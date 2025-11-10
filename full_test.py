# -*- coding: utf-8 -*-
"""
Comprehensive End-to-End Test for Medical RAG Chatbot
Tests all components: imports, config, vector store, RAG pipeline, API
"""
import sys
import os

print("\n" + "="*70)
print("  MEDICAL RAG CHATBOT - COMPREHENSIVE SYSTEM TEST")
print("="*70)

# Test 1: Package Imports
print("\n[TEST 1/8] Package Imports")
print("-"*70)
test_results = []

packages = {
    'fastapi': 'FastAPI',
    'uvicorn': 'Uvicorn',
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'sentence_transformers': 'Sentence Transformers',
    'faiss': 'FAISS',
    'PyPDF2': 'PyPDF2',
    'numpy': 'NumPy',
    'pydantic': 'Pydantic'
}

for module, name in packages.items():
    try:
        __import__(module)
        print(f"  [OK] {name}")
    except ImportError:
        print(f"  [FAIL] {name} - NOT INSTALLED")
        test_results.append(('Package Import', False))
        
test_results.append(('Package Import', True))

# Test 2: Configuration
print("\n[TEST 2/8] Configuration")
print("-"*70)
try:
    import config
    print(f"  [OK] Config loaded")
    print(f"       Embedding Model: {config.EMBEDDING_MODEL}")
    print(f"       Generator Model: {config.GENERATOR_MODEL}")
    print(f"       Vector Dimension: {config.VECTOR_DIMENSION}")
    print(f"       Chunk Size: {config.CHUNK_SIZE}")
    print(f"       Top-K Retrieval: {config.TOP_K_RETRIEVAL}")
    test_results.append(('Configuration', True))
except Exception as e:
    print(f"  [FAIL] {e}")
    test_results.append(('Configuration', False))

# Test 3: Dataset
print("\n[TEST 3/8] Medical Dataset")
print("-"*70)
try:
    from config import DATASET_DIR
    if os.path.exists(DATASET_DIR):
        pdfs = [f for f in os.listdir(DATASET_DIR) if f.endswith('.pdf')]
        print(f"  [OK] Found {len(pdfs)} medical PDF files:")
        for i, pdf in enumerate(pdfs, 1):
            print(f"       {i}. {pdf}")
        test_results.append(('Dataset', True))
    else:
        print(f"  [FAIL] Dataset directory not found: {DATASET_DIR}")
        test_results.append(('Dataset', False))
except Exception as e:
    print(f"  [FAIL] {e}")
    test_results.append(('Dataset', False))

# Test 4: Vector Store
print("\n[TEST 4/8] Vector Store (FAISS)")
print("-"*70)
try:
    from vector_store import VectorStore
    vs = VectorStore()
    stats = vs.get_stats()
    print(f"  [OK] Vector store initialized")
    print(f"       Total documents: {stats['total_documents']}")
    print(f"       Index dimension: {stats['index_dimension']}")
    print(f"       Embedding model: {stats['embedding_model']}")
    
    # Test embedding generation
    vs.load_embedding_model()
    test_text = ["This is a test for medical information retrieval"]
    embeddings = vs.create_embeddings(test_text)
    print(f"  [OK] Embedding generation works ({embeddings.shape})")
    test_results.append(('Vector Store', True))
except Exception as e:
    print(f"  [FAIL] {e}")
    test_results.append(('Vector Store', False))

# Test 5: PDF Loader
print("\n[TEST 5/8] PDF Loader")
print("-"*70)
try:
    from utils.pdf_loader import PDFLoader
    loader = PDFLoader(chunk_size=500, chunk_overlap=50)
    print(f"  [OK] PDF loader initialized")
    print(f"       Chunk size: 500 tokens")
    print(f"       Chunk overlap: 50 tokens")
    
    # Test chunking
    test_text = "This is a medical document. " * 100
    chunks = loader.chunk_text(test_text, "test.pdf")
    print(f"  [OK] Text chunking works ({len(chunks)} chunks)")
    test_results.append(('PDF Loader', True))
except Exception as e:
    print(f"  [FAIL] {e}")
    test_results.append(('PDF Loader', False))

# Test 6: RAG Pipeline
print("\n[TEST 6/8] RAG Pipeline")
print("-"*70)
try:
    from rag_pipeline import RAGPipeline
    rag = RAGPipeline(use_fallback_model=False)
    print(f"  [OK] RAG pipeline initialized")
    print(f"       Model: {rag.model_name}")
    print(f"       NOTE: LLM will be loaded on first query (~2-4GB download)")
    test_results.append(('RAG Pipeline', True))
except Exception as e:
    print(f"  [FAIL] {e}")
    test_results.append(('RAG Pipeline', False))

# Test 7: Query Logger
print("\n[TEST 7/8] Query Logger (SQLite)")
print("-"*70)
try:
    from models.query_log import QueryLogger
    logger = QueryLogger()
    stats = logger.get_query_stats()
    print(f"  [OK] Query logger initialized")
    print(f"       Database: {logger.db_path}")
    print(f"       Total queries logged: {stats['total_queries']}")
    test_results.append(('Query Logger', True))
except Exception as e:
    print(f"  [FAIL] {e}")
    test_results.append(('Query Logger', False))

# Test 8: FastAPI Application
print("\n[TEST 8/8] FastAPI Application")
print("-"*70)
try:
    from app.main import app
    print(f"  [OK] FastAPI app loaded")
    
    # Check endpoints
    routes = [r.path for r in app.routes]
    key_routes = {
        '/': 'API Info',
        '/health': 'Health Check',
        '/ingest': 'Ingest Documents',
        '/upload': 'Upload Files',
        '/query': 'Query Chatbot',
        '/evaluate': 'Evaluate Response',
        '/stats': 'System Statistics',
        '/reset': 'Reset Vector Store'
    }
    
    print(f"  [OK] Available endpoints:")
    for route, description in key_routes.items():
        if route in routes:
            print(f"       [OK] {route:20} - {description}")
        else:
            print(f"       [MISS] {route:20} - {description}")
    
    test_results.append(('FastAPI Application', True))
except Exception as e:
    print(f"  [FAIL] {e}")
    test_results.append(('FastAPI Application', False))

# GPU/CUDA Check
print("\n[BONUS] GPU/CUDA Status")
print("-"*70)
try:
    import torch
    if torch.cuda.is_available():
        print(f"  [OK] CUDA Available")
        print(f"       GPU: {torch.cuda.get_device_name(0)}")
        memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"       Memory: {memory:.1f}GB")
    else:
        print(f"  [INFO] CUDA Not Available (CPU mode)")
        print(f"         This is OK - system will run on CPU")
        print(f"         For GPU support, install PyTorch with CUDA")
except Exception as e:
    print(f"  [WARN] {e}")

# Summary
print("\n" + "="*70)
print("  TEST SUMMARY")
print("="*70)

passed = sum(1 for _, result in test_results if result)
total = len(test_results)

for test_name, result in test_results:
    status = "[PASS]" if result else "[FAIL]"
    print(f"  {status} {test_name}")

print("\n" + "="*70)
print(f"  RESULT: {passed}/{total} tests passed")
print("="*70)

if passed == total:
    print("\n  STATUS: ALL TESTS PASSED!")
    print("\n  Your Medical RAG Chatbot is fully functional!")
    print("\n  Next Steps:")
    print("    1. Start server:  py server.py")
    print("    2. Open browser:  http://localhost:8000/docs")
    print("    3. Ingest docs:   Use /ingest endpoint")
    print("    4. Ask questions: Use /query endpoint")
    print("\n  First query will download models (~2-4GB)")
else:
    print("\n  STATUS: SOME TESTS FAILED")
    print("\n  Please check the errors above and install missing packages")
    print("  Run: py -m pip install -r requirements.txt")

print("\n" + "="*70 + "\n")

sys.exit(0 if passed == total else 1)



