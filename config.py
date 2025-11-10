"""
Configuration file for Medical RAG Chatbot
"""
import os

# Model Configuration
# Optimized for RTX 2050 (4GB VRAM) with Medical Specialization
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
GENERATOR_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small, fast, works on any GPU
FALLBACK_GENERATOR_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Same as primary
USE_GPTQ = False  # TinyLlama doesn't need quantization

# Vector Store Configuration
VECTOR_DIMENSION = 768  # dimension for bge-base-en-v1.5
FAISS_INDEX_PATH = "data/faiss_index"
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50  # tokens for context preservation

# Retrieval Configuration
TOP_K_RETRIEVAL = 5  # number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.5  # minimum similarity score

# Generation Configuration
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

# Database Configuration
DATABASE_PATH = "data/query_logs.db"
USE_MONGODB = False  # Set to True to use MongoDB instead of SQLite
MONGODB_URI = "mongodb://localhost:27017/"
MONGODB_DB_NAME = "medical_rag"

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Directories
DATA_DIR = "data"
UPLOAD_DIR = "data/uploads"
DATASET_DIR = "HackACure-Dataset/Dataset"

# Create necessary directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

# Prompt Template
RAG_PROMPT_TEMPLATE = """You are a helpful and trustworthy medical assistant with expertise in various medical domains.
Use the context below to answer the question accurately. Always cite your sources and be explicit about the medical domain.

Context:
{context}

Question: {question}

Provide a detailed, accurate answer based on the context above. If the context doesn't contain enough information, say so.
Include specific citations to the source documents used."""

SYSTEM_PROMPT = """You are a medical AI assistant. Your role is to provide accurate, evidence-based medical information.
Always cite your sources and acknowledge the limits of your knowledge."""



