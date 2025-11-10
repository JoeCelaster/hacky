"""Direct ingestion - bypassing server completely"""
import os
import sys

print("\n" + "="*70)
print("  DIRECT DATABASE INGESTION")
print("="*70)

print("\n[1/4] Loading components...")
from utils.pdf_loader import PDFLoader
from vector_store import get_vector_store
from config import DATASET_DIR

print("[2/4] Processing PDFs from:", DATASET_DIR)
loader = PDFLoader()
chunks = loader.load_directory(DATASET_DIR)
print("Generated", len(chunks), "chunks")

print("[3/4] Creating vector database...")
vs = get_vector_store()
vs.add_documents(chunks)
vs.save_index()

print("[4/4] Verifying...")
stats = vs.get_stats()
print("SUCCESS! Stored", stats['total_documents'], "chunks")
print("Sources:", stats['sources'])
print("\n" + "="*70)
print("DONE! Documents are now in the database!")
print("="*70 + "\n")
