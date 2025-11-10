# -*- coding: utf-8 -*-
"""Quick system test"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("\n" + "="*60)
print("  Testing Medical RAG Chatbot System")
print("="*60)

# Test 1: Check imports
print("\n[1/5] Testing imports...")
try:
    from app.main import app
    print("  OK - API loaded")
except Exception as e:
    print(f"  FAIL - {e}")
    exit(1)

# Test 2: Check endpoints
print("\n[2/5] Checking endpoints...")
routes = [r.path for r in app.routes]
key_routes = ['/', '/health', '/ingest', '/query', '/upload', '/evaluate', '/stats', '/reset']
found = [r for r in routes if r in key_routes]
print(f"  Found {len(found)}/{len(key_routes)} key endpoints")
for route in found:
    print(f"    - {route}")

# Test 3: Check config
print("\n[3/5] Checking configuration...")
try:
    import config
    print(f"  Embedding: {config.EMBEDDING_MODEL}")
    print(f"  Generator: {config.GENERATOR_MODEL}")
    print(f"  Vector Dim: {config.VECTOR_DIMENSION}")
except Exception as e:
    print(f"  FAIL - {e}")

# Test 4: Check GPU
print("\n[4/5] Checking GPU...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  CUDA: Available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  CUDA: Not available (CPU mode)")
except Exception as e:
    print(f"  ERROR - {e}")

# Test 5: Check dataset
print("\n[5/5] Checking dataset...")
try:
    import os
    from config import DATASET_DIR
    if os.path.exists(DATASET_DIR):
        pdfs = [f for f in os.listdir(DATASET_DIR) if f.endswith('.pdf')]
        print(f"  Found {len(pdfs)} PDF files")
        for pdf in pdfs[:3]:
            print(f"    - {pdf}")
        if len(pdfs) > 3:
            print(f"    ... and {len(pdfs)-3} more")
    else:
        print(f"  Dataset not found at: {DATASET_DIR}")
except Exception as e:
    print(f"  ERROR - {e}")

print("\n" + "="*60)
print("  System Check Complete!")
print("="*60)
print("\nTo start the server:")
print("  py server.py")
print("\nThen open:")
print("  http://localhost:8000/docs")
print()
