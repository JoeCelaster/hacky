"""
Quick Test Script for Medical RAG Chatbot
Tests basic functionality without requiring full setup
"""
import sys
import os

def test_imports():
    """Test if all required packages are installed"""
    print("\n" + "="*60)
    print("🔍 Testing Package Imports...")
    print("="*60)
    
    packages = {
        'FastAPI': 'fastapi',
        'Uvicorn': 'uvicorn',
        'PyTorch': 'torch',
        'Transformers': 'transformers',
        'Sentence Transformers': 'sentence_transformers',
        'FAISS': 'faiss',
        'LangChain': 'langchain',
        'PyPDF2': 'PyPDF2',
    }
    
    results = {}
    for name, package in packages.items():
        try:
            __import__(package)
            print(f"✓ {name:25} - OK")
            results[name] = True
        except ImportError as e:
            print(f"✗ {name:25} - FAILED")
            results[name] = False
    
    return all(results.values())

def test_cuda():
    """Test CUDA availability"""
    print("\n" + "="*60)
    print("🎮 Testing GPU/CUDA...")
    print("="*60)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✓ CUDA Available: Yes")
            print(f"  GPU: {gpu_name}")
            print(f"  Memory: {gpu_memory:.1f}GB")
        else:
            print(f"⚠️ CUDA Available: No (will use CPU)")
            print(f"  This is OK - models will run on CPU")
        return True
    except Exception as e:
        print(f"✗ Error testing CUDA: {e}")
        return False

def test_config():
    """Test configuration file"""
    print("\n" + "="*60)
    print("⚙️ Testing Configuration...")
    print("="*60)
    
    try:
        import config
        print(f"✓ Embedding Model: {config.EMBEDDING_MODEL}")
        print(f"✓ Generator Model: {config.GENERATOR_MODEL}")
        print(f"✓ Vector Dimension: {config.VECTOR_DIMENSION}")
        print(f"✓ Dataset Dir: {config.DATASET_DIR}")
        
        # Check if dataset exists
        if os.path.exists(config.DATASET_DIR):
            pdf_files = [f for f in os.listdir(config.DATASET_DIR) if f.endswith('.pdf')]
            print(f"✓ Dataset Found: {len(pdf_files)} PDF files")
        else:
            print(f"⚠️ Dataset directory not found at: {config.DATASET_DIR}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_basic_functionality():
    """Test basic RAG components"""
    print("\n" + "="*60)
    print("🧪 Testing Basic Functionality...")
    print("="*60)
    
    try:
        # Test embedding model loading
        print("\n1. Testing Embedding Model...")
        from sentence_transformers import SentenceTransformer
        from config import EMBEDDING_MODEL
        
        print(f"   Loading: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Test embedding generation
        test_text = "This is a test sentence about medical information."
        embedding = model.encode(test_text)
        print(f"   ✓ Embedding generated: {len(embedding)} dimensions")
        
        # Test vector store
        print("\n2. Testing Vector Store...")
        from vector_store import VectorStore
        vs = VectorStore()
        stats = vs.get_stats()
        print(f"   ✓ Vector store initialized")
        print(f"   Documents: {stats['total_documents']}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_api_structure():
    """Test FastAPI application structure"""
    print("\n" + "="*60)
    print("🌐 Testing API Structure...")
    print("="*60)
    
    try:
        from app.main import app
        
        routes = [route.path for route in app.routes]
        expected_routes = ['/', '/health', '/ingest', '/query', '/upload']
        
        for route in expected_routes:
            if route in routes:
                print(f"✓ Endpoint found: {route}")
            else:
                print(f"⚠️ Endpoint missing: {route}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading API: {e}")
        return False

def main():
    print("""
============================================================
  Medical RAG Chatbot - Quick Test Suite
============================================================
    """)
    
    results = []
    
    # Run tests
    results.append(("Package Imports", test_imports()))
    results.append(("GPU/CUDA", test_cuda()))
    results.append(("Configuration", test_config()))
    results.append(("API Structure", test_api_structure()))
    results.append(("Basic Functionality", test_basic_functionality()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your system is ready!")
        print("\n📝 Next Steps:")
        print("  1. Start server: py server.py")
        print("  2. Open browser: http://localhost:8000/docs")
        print("  3. Run full test: py test_system.py")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
        print("  Try running: py quick_install.py")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

