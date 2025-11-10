"""
Core RAG Pipeline: Retrieval, Generation, and Citation
"""
import time
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from vector_store import get_vector_store
from config import (
    GENERATOR_MODEL,
    FALLBACK_GENERATOR_MODEL,
    RAG_PROMPT_TEMPLATE,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    TOP_K_RETRIEVAL,
    SIMILARITY_THRESHOLD
)

# Try importing GPTQ support
try:
    from auto_gptq import AutoGPTQForCausalLM
    GPTQ_AVAILABLE = True
except ImportError:
    GPTQ_AVAILABLE = False
    print("[INFO] auto-gptq not installed. GPTQ models will use standard loading.")


class RAGPipeline:
    """
    Complete RAG pipeline for medical Q&A
    """
    
    def __init__(self, use_fallback_model: bool = False):
        """
        Initialize RAG Pipeline
        
        Args:
            use_fallback_model: Use Mistral instead of BiomedGPT if True
        """
        self.vector_store = get_vector_store()
        self.generator = None
        self.tokenizer = None
        self.model_name = FALLBACK_GENERATOR_MODEL if use_fallback_model else GENERATOR_MODEL
        self.use_fallback = use_fallback_model
        
    def load_generator_model(self):
        """Load the language model for text generation"""
        if self.generator is not None:
            return
        
        print(f"Loading generator model: {self.model_name}")
        print("This may take a few minutes on first run...")
        
        try:
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            if device == "cuda":
                # Check available VRAM
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"GPU Memory: {total_memory:.1f}GB")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present (for some models)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Check if this is a GPTQ quantized model
            is_gptq_model = "GPTQ" in self.model_name or "gptq" in self.model_name
            
            if is_gptq_model and GPTQ_AVAILABLE and device == "cuda":
                # Load GPTQ quantized model (optimized for 4GB VRAM)
                print("[INFO] Loading GPTQ quantized model...")
                model = AutoGPTQForCausalLM.from_quantized(
                    self.model_name,
                    device="cuda:0",
                    use_triton=False,  # More compatible
                    use_safetensors=True,
                    trust_remote_code=True,
                    inject_fused_attention=False,
                    inject_fused_mlp=False,
                )
            else:
                # Load standard model with memory-optimized settings
                model_kwargs = {
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                    "device_map": "auto"
                }
                
                if device == "cuda":
                    # Use float16 for GPU to save memory
                    model_kwargs["torch_dtype"] = torch.float16
                else:
                    model_kwargs["torch_dtype"] = torch.float32
                
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                device=0 if device == "cuda" else -1
            )
            
            print("[OK] Generator model loaded successfully")
            
            if device == "cuda":
                allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
                print(f"[OK] GPU Memory Used: {allocated_memory:.2f}GB")
            
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            if not self.use_fallback and "BiomedGPT" in self.model_name:
                print("Falling back to Mistral model...")
                self.model_name = FALLBACK_GENERATOR_MODEL
                self.use_fallback = True
                self.load_generator_model()
            else:
                raise
    
    def retrieve_context(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        """
        Retrieve relevant documents from vector store
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        # Load vector store if not already loaded
        if self.vector_store.index is None:
            self.vector_store.load_index()
        
        # Retrieve documents
        results = self.vector_store.search(query, top_k=top_k)
        
        # Filter by similarity threshold
        filtered_results = [
            doc for doc in results
            if doc['similarity_score'] >= SIMILARITY_THRESHOLD
        ]
        
        return filtered_results
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for idx, doc in enumerate(retrieved_docs, 1):
            context_parts.append(
                f"[Source {idx}: {doc['source']}]\n{doc['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def extract_citations(self, retrieved_docs: List[Dict]) -> List[str]:
        """
        Extract unique citations from retrieved documents
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            List of unique source citations
        """
        sources = list(set([doc['source'] for doc in retrieved_docs]))
        return sorted(sources)
    
    def generate_reasoning_summary(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        Generate reasoning summary explaining why these sources were chosen
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            
        Returns:
            Reasoning summary string
        """
        if not retrieved_docs:
            return "No relevant documents found in the knowledge base."
        
        # Get top sources with scores
        top_sources = []
        for doc in retrieved_docs[:3]:  # Top 3 sources
            top_sources.append(
                f"{doc['source']} (relevance: {doc['similarity_score']:.2f})"
            )
        
        reasoning = f"Retrieved {len(retrieved_docs)} relevant document(s) from: {', '.join(top_sources)}. "
        reasoning += "These sources were selected based on semantic similarity to your query about medical information."
        
        return reasoning
    
    def generate_answer(
        self,
        query: str,
        context: str,
        max_tokens: int = MAX_NEW_TOKENS
    ) -> str:
        """
        Generate answer using the language model
        
        Args:
            query: User query
            context: Retrieved context
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated answer
        """
        # Load model if not already loaded
        if self.generator is None:
            self.load_generator_model()
        
        # Format prompt
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=query
        )
        
        # Generate response
        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=max_tokens,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                return_full_text=False
            )
            
            answer = outputs[0]['generated_text'].strip()
            
            # Clean up the answer if needed
            if not answer:
                answer = "I apologize, but I couldn't generate a proper response based on the available context."
            
            return answer
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def query(
        self,
        question: str,
        top_k: int = TOP_K_RETRIEVAL
    ) -> Dict:
        """
        Complete RAG query pipeline
        
        Args:
            question: User's medical question
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer, citations, and reasoning
        """
        start_time = time.time()
        
        # Step 1: Retrieve relevant documents
        print(f"\n[1/3] Retrieving relevant documents for: {question[:100]}...")
        retrieved_docs = self.retrieve_context(question, top_k=top_k)
        
        if not retrieved_docs:
            return {
                'answer': "I couldn't find relevant information in the knowledge base to answer your question. Please ensure the medical documents have been ingested.",
                'citations': [],
                'reasoning_summary': "No relevant documents found.",
                'num_retrieved': 0,
                'response_time_ms': (time.time() - start_time) * 1000
            }
        
        print(f"[OK] Retrieved {len(retrieved_docs)} relevant documents")
        
        # Step 2: Format context
        context = self.format_context(retrieved_docs)
        
        # Step 3: Generate answer
        print("[2/3] Generating answer...")
        answer = self.generate_answer(question, context)
        
        # Step 4: Extract citations
        citations = self.extract_citations(retrieved_docs)
        
        # Step 5: Generate reasoning summary
        reasoning = self.generate_reasoning_summary(question, retrieved_docs)
        
        response_time = (time.time() - start_time) * 1000
        
        print(f"[OK] Answer generated in {response_time:.2f}ms")
        
        return {
            'answer': answer,
            'citations': citations,
            'reasoning_summary': reasoning,
            'num_retrieved': len(retrieved_docs),
            'response_time_ms': response_time,
            'retrieved_docs': retrieved_docs  # For evaluation purposes
        }
    
    def evaluate_response(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Dict]
    ) -> Dict:
        """
        Simple evaluation metrics (placeholder for RAGAS)
        
        Args:
            query: User query
            answer: Generated answer
            retrieved_docs: Retrieved documents
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Basic heuristic-based evaluation
        # In production, use RAGAS library for proper evaluation
        
        metrics = {
            'faithfulness_score': 0.0,
            'context_recall': 0.0,
            'context_precision': 0.0,
            'answer_relevancy': 0.0
        }
        
        # Simple faithfulness: Check if answer mentions sources
        if any(doc['source'].split('.')[0].lower() in answer.lower() for doc in retrieved_docs):
            metrics['faithfulness_score'] = 0.8
        else:
            metrics['faithfulness_score'] = 0.5
        
        # Context recall: Ratio of retrieved docs to query terms
        query_terms = set(query.lower().split())
        context_terms = set()
        for doc in retrieved_docs:
            context_terms.update(doc['text'].lower().split())
        
        overlap = len(query_terms.intersection(context_terms))
        metrics['context_recall'] = min(overlap / len(query_terms), 1.0) if query_terms else 0.0
        
        # Context precision: Simple heuristic based on similarity scores
        if retrieved_docs:
            avg_similarity = sum(doc['similarity_score'] for doc in retrieved_docs) / len(retrieved_docs)
            metrics['context_precision'] = avg_similarity
        
        # Answer relevancy: Check if answer is substantial
        if len(answer.split()) > 20 and "Error" not in answer:
            metrics['answer_relevancy'] = 0.8
        else:
            metrics['answer_relevancy'] = 0.4
        
        return metrics


# Singleton instance
_rag_pipeline = None


def get_rag_pipeline(use_fallback: bool = False) -> RAGPipeline:
    """Get or create the global RAG pipeline instance"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline(use_fallback_model=use_fallback)
    return _rag_pipeline



