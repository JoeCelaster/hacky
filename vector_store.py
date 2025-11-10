"""
FAISS Vector Store Management for Medical RAG System
"""
import os
import json
import pickle
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import (
    EMBEDDING_MODEL,
    VECTOR_DIMENSION,
    FAISS_INDEX_PATH,
    TOP_K_RETRIEVAL
)


class VectorStore:
    """
    Manages FAISS vector store for document embeddings
    """
    
    def __init__(self, index_path: str = FAISS_INDEX_PATH):
        """
        Initialize Vector Store
        
        Args:
            index_path: Directory path to store FAISS index
        """
        self.index_path = index_path
        self.index = None
        self.documents = []  # Store original documents with metadata
        self.embedding_model = None
        
        # Ensure index directory exists
        os.makedirs(self.index_path, exist_ok=True)
        
    def load_embedding_model(self):
        """Load the sentence transformer embedding model"""
        if self.embedding_model is None:
            print(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            print("Embedding model loaded successfully")
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        if self.embedding_model is None:
            self.load_embedding_model()
        
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        return embeddings
    
    def create_index(self, dimension: int = VECTOR_DIMENSION):
        """
        Create a new FAISS index
        
        Args:
            dimension: Dimension of the embeddings
        """
        # Use IndexFlatL2 for exact search (good for small to medium datasets)
        # For larger datasets, consider IndexIVFFlat or IndexHNSWFlat
        self.index = faiss.IndexFlatL2(dimension)
        print(f"Created FAISS index with dimension {dimension}")
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Add documents to the vector store
        
        Args:
            documents: List of document dictionaries with 'text', 'source', 'chunk_id'
        """
        if not documents:
            print("No documents to add")
            return
        
        # Extract texts
        texts = [doc['text'] for doc in documents]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.create_embeddings(texts)
        
        # Create index if it doesn't exist
        if self.index is None:
            self.create_index(embeddings.shape[1])
        
        # Add embeddings to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents
        self.documents.extend(documents)
        
        print(f"Added {len(documents)} documents to vector store")
        print(f"Total documents in store: {len(self.documents)}")
    
    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of documents with similarity scores
        """
        if self.index is None or len(self.documents) == 0:
            print("Vector store is empty. Please ingest documents first.")
            return []
        
        # Generate query embedding
        query_embedding = self.create_embeddings([query])
        
        # Search in FAISS
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            min(top_k, len(self.documents))
        )
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                # Convert L2 distance to similarity score (inverse)
                doc['similarity_score'] = float(1 / (1 + dist))
                results.append(doc)
        
        return results
    
    def save_index(self):
        """Save FAISS index and documents to disk"""
        if self.index is None:
            print("No index to save")
            return
        
        # Save FAISS index
        index_file = os.path.join(self.index_path, "faiss_index.bin")
        faiss.write_index(self.index, index_file)
        
        # Save documents metadata
        docs_file = os.path.join(self.index_path, "documents.pkl")
        with open(docs_file, 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save index info
        info_file = os.path.join(self.index_path, "index_info.json")
        with open(info_file, 'w') as f:
            json.dump({
                'num_documents': len(self.documents),
                'dimension': self.index.d,
                'embedding_model': EMBEDDING_MODEL
            }, f, indent=2)
        
        print(f"Index saved to {self.index_path}")
    
    def load_index(self):
        """Load FAISS index and documents from disk"""
        index_file = os.path.join(self.index_path, "faiss_index.bin")
        docs_file = os.path.join(self.index_path, "documents.pkl")
        
        if not os.path.exists(index_file) or not os.path.exists(docs_file):
            print("No saved index found. Please ingest documents first.")
            return False
        
        # Load FAISS index
        self.index = faiss.read_index(index_file)
        
        # Load documents
        with open(docs_file, 'rb') as f:
            self.documents = pickle.load(f)
        
        # Load embedding model
        self.load_embedding_model()
        
        print(f"Loaded index with {len(self.documents)} documents")
        return True
    
    def clear_index(self):
        """Clear the current index"""
        self.index = None
        self.documents = []
        print("Index cleared")
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store
        
        Returns:
            Dictionary with statistics
        """
        if self.index is None:
            return {
                'total_documents': 0,
                'index_dimension': 0,
                'sources': [],
                'embedding_model': EMBEDDING_MODEL
            }
        
        sources = list(set([doc['source'] for doc in self.documents]))
        
        return {
            'total_documents': len(self.documents),
            'index_dimension': self.index.d if self.index else 0,
            'sources': sources,
            'embedding_model': EMBEDDING_MODEL
        }


# Singleton instance
_vector_store = None


def get_vector_store() -> VectorStore:
    """Get or create the global vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store




