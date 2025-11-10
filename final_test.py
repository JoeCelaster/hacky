# -*- coding: utf-8 -*-
"""Final clean test - No Unicode, direct ingestion"""
import sys
import io

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("\n" + "="*70)
print("  FINAL TEST - DIRECT INGESTION")
print("="*70)

# Step 1: Direct ingestion without API
print("\n[STEP 1] Loading components...")
try:
    from utils.pdf_loader import PDFLoader
    from vector_store import get_vector_store  
    from config import DATASET_DIR
    print("[OK] All imports successful")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

# Step 2: Load and chunk PDFs
print(f"\n[STEP 2] Processing PDFs from: {DATASET_DIR}")
try:
    loader = PDFLoader()
    chunks = loader.load_directory(DATASET_DIR)
    print(f"[OK] Generated {len(chunks)} total chunks")
except Exception as e:
    print(f"[ERROR] PDF loading failed: {e}")
    sys.exit(1)

# Step 3: Add to vector store
print(f"\n[STEP 3] Building vector database...")
try:
    vs = get_vector_store()
    vs.add_documents(chunks)
    vs.save_index()
    print("[OK] Vector store created and saved")
except Exception as e:
    print(f"[ERROR] Vector store failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Verify
print(f"\n[STEP 4] Verifying...")
stats = vs.get_stats()
print(f"[OK] Total documents: {stats['total_documents']}")
print(f"[OK] Sources: {len(stats['sources'])}")
for i, source in enumerate(stats['sources'], 1):
    print(f"     {i}. {source}")

print("\n" + "="*70)
print("  SUCCESS! Ingestion complete!")
print("="*70)
print("\nNow you can query the system!")
print("="*70 + "\n")



