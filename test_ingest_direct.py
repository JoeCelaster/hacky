# -*- coding: utf-8 -*-
"""Direct ingestion test without server"""
import sys
sys.path.insert(0, '.')

print("Testing direct ingestion...")

try:
    from utils.pdf_loader import PDFLoader
    from vector_store import get_vector_store
    from config import DATASET_DIR
    
    print(f"[1/3] Loading PDF loader...")
    pdf_loader = PDFLoader()
    
    print(f"[2/3] Loading documents from: {DATASET_DIR}")
    chunks = pdf_loader.load_directory(DATASET_DIR)
    print(f"[OK] Generated {len(chunks)} chunks from documents")
    
    print(f"[3/3] Adding to vector store...")
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    vector_store.save_index()
    
    print(f"\n[SUCCESS] Ingestion complete!")
    print(f"  Total chunks: {len(chunks)}")
    
    stats = vector_store.get_stats()
    print(f"  Sources: {', '.join(stats['sources'])}")
    
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()



