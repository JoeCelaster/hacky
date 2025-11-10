"""
FastAPI Application for Medical RAG Chatbot
"""
import os
import time
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from vector_store import get_vector_store
from rag_pipeline import get_rag_pipeline
from models.query_log import get_query_logger
from utils.pdf_loader import PDFLoader
from config import (
    API_HOST,
    API_PORT,
    UPLOAD_DIR,
    DATASET_DIR,
    TOP_K_RETRIEVAL
)


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    question: str = Field(..., description="Medical question to answer")
    top_k: Optional[int] = Field(TOP_K_RETRIEVAL, description="Number of documents to retrieve")


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str
    citations: List[str]
    reasoning_summary: str
    num_retrieved: int
    response_time_ms: float


class IngestRequest(BaseModel):
    """Request model for ingesting from directory"""
    directory_path: Optional[str] = Field(None, description="Path to directory containing documents")
    use_default_dataset: bool = Field(True, description="Use the default HackACure dataset")


class IngestResponse(BaseModel):
    """Response model for ingest endpoint"""
    status: str
    num_documents: int
    num_chunks: int
    sources: List[str]
    message: str


class EvaluationResponse(BaseModel):
    """Response model for evaluation endpoint"""
    query: str
    faithfulness_score: float
    context_recall: float
    context_precision: float
    answer_relevancy: float


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    vector_store_ready: bool
    total_documents: int
    model_loaded: bool


# Initialize FastAPI app
app = FastAPI(
    title="Medical RAG Chatbot API",
    description="Retrieval-Augmented Generation API for Medical Q&A with citations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
vector_store = get_vector_store()
rag_pipeline = None  # Lazy load
query_logger = get_query_logger()


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("[STARTUP] Medical RAG Chatbot API starting...")
    
    # Try to load existing vector store
    if vector_store.load_index():
        print("[OK] Loaded existing vector store")
    else:
        print("[INFO] No existing vector store found. Use /ingest to add documents.")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Medical RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "ingest": "/ingest",
            "upload": "/upload",
            "query": "/query",
            "evaluate": "/evaluate",
            "stats": "/stats"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint
    Returns the status of the API and vector store
    """
    global rag_pipeline
    
    stats = vector_store.get_stats()
    
    return HealthResponse(
        status="healthy",
        vector_store_ready=stats['total_documents'] > 0,
        total_documents=stats['total_documents'],
        model_loaded=rag_pipeline is not None and rag_pipeline.generator is not None
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Data Ingestion"])
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents from a directory or use default dataset
    Supports .pdf and .txt files
    """
    try:
        pdf_loader = PDFLoader()
        
        # Determine directory to process
        if request.use_default_dataset:
            directory = DATASET_DIR
            print(f"[INFO] Using default dataset directory: {directory}")
        elif request.directory_path:
            directory = request.directory_path
            if not os.path.exists(directory):
                raise HTTPException(status_code=404, detail="Directory not found")
        else:
            raise HTTPException(
                status_code=400,
                detail="Either use_default_dataset must be True or directory_path must be provided"
            )
        
        # Load and chunk documents
        print(f"[INFO] Loading documents from: {directory}")
        chunks = pdf_loader.load_directory(directory)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No documents found or no text could be extracted"
            )
        
        # Add to vector store
        print(f"[INFO] Adding {len(chunks)} chunks to vector store...")
        vector_store.add_documents(chunks)
        
        # Save index
        vector_store.save_index()
        
        # Get sources
        sources = list(set([chunk['source'] for chunk in chunks]))
        
        return IngestResponse(
            status="success",
            num_documents=len(sources),
            num_chunks=len(chunks),
            sources=sources,
            message=f"Successfully ingested {len(sources)} documents with {len(chunks)} chunks"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=IngestResponse, tags=["Data Ingestion"])
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and ingest documents (.pdf or .txt files)
    """
    try:
        pdf_loader = PDFLoader()
        all_chunks = []
        sources = []
        
        for file in files:
            # Validate file type
            if not file.filename.endswith(('.pdf', '.txt')):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.filename}. Only .pdf and .txt are supported."
                )
            
            # Save uploaded file
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, 'wb') as f:
                content = await file.read()
                f.write(content)
            
            # Process file
            print(f"[INFO] Processing: {file.filename}")
            chunks = pdf_loader.load_and_chunk_document(file_path)
            all_chunks.extend(chunks)
            sources.append(file.filename)
        
        if not all_chunks:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from uploaded files"
            )
        
        # Add to vector store
        vector_store.add_documents(all_chunks)
        vector_store.save_index()
        
        return IngestResponse(
            status="success",
            num_documents=len(sources),
            num_chunks=len(all_chunks),
            sources=sources,
            message=f"Successfully uploaded and ingested {len(sources)} documents"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_chatbot(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Query the medical chatbot
    Returns answer with citations and reasoning
    """
    global rag_pipeline
    
    try:
        # Check if vector store is ready
        if vector_store.index is None or len(vector_store.documents) == 0:
            raise HTTPException(
                status_code=400,
                detail="Vector store is empty. Please ingest documents first using /ingest or /upload"
            )
        
        # Initialize RAG pipeline if not already done
        if rag_pipeline is None:
            print("[INFO] Initializing RAG pipeline (first query)...")
            rag_pipeline = get_rag_pipeline(use_fallback=False)
        
        # Execute query
        result = rag_pipeline.query(
            question=request.question,
            top_k=request.top_k
        )
        
        # Log query in background
        def log_query_task():
            query_logger.log_query(
                query_text=request.question,
                answer=result['answer'],
                citations=result['citations'],
                reasoning_summary=result['reasoning_summary'],
                top_k=request.top_k,
                response_time_ms=result['response_time_ms'],
                num_retrieved_docs=result['num_retrieved']
            )
        
        background_tasks.add_task(log_query_task)
        
        return QueryResponse(
            answer=result['answer'],
            citations=result['citations'],
            reasoning_summary=result['reasoning_summary'],
            num_retrieved=result['num_retrieved'],
            response_time_ms=result['response_time_ms']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
async def evaluate_query(request: QueryRequest):
    """
    Evaluate a query using RAGAS-like metrics
    Returns faithfulness, context recall, precision, and answer relevancy
    """
    global rag_pipeline
    
    try:
        # Check if vector store is ready
        if vector_store.index is None:
            raise HTTPException(
                status_code=400,
                detail="Vector store is empty. Please ingest documents first."
            )
        
        # Initialize RAG pipeline if needed
        if rag_pipeline is None:
            rag_pipeline = get_rag_pipeline(use_fallback=False)
        
        # Execute query
        result = rag_pipeline.query(
            question=request.question,
            top_k=request.top_k
        )
        
        # Evaluate response
        metrics = rag_pipeline.evaluate_response(
            query=request.question,
            answer=result['answer'],
            retrieved_docs=result.get('retrieved_docs', [])
        )
        
        return EvaluationResponse(
            query=request.question,
            faithfulness_score=metrics['faithfulness_score'],
            context_recall=metrics['context_recall'],
            context_precision=metrics['context_precision'],
            answer_relevancy=metrics['answer_relevancy']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["General"])
async def get_statistics():
    """
    Get statistics about the system
    Returns vector store stats and query logs
    """
    try:
        vector_stats = vector_store.get_stats()
        query_stats = query_logger.get_query_stats()
        recent_queries = query_logger.get_recent_queries(limit=5)
        
        return {
            "vector_store": vector_stats,
            "query_logs": query_stats,
            "recent_queries": recent_queries
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/reset", tags=["Admin"])
async def reset_vector_store():
    """
    Reset the vector store (admin endpoint)
    Clears all ingested documents
    """
    try:
        vector_store.clear_index()
        return {
            "status": "success",
            "message": "Vector store has been reset. Please ingest new documents."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         Medical RAG Chatbot API - Starting...            ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    )




