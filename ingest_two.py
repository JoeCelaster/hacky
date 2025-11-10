"""Ingest just 2 documents for quick testing"""
import os

print("\n" + "="*70)
print("  INGESTING 2 DOCUMENTS (TEST)")
print("="*70)

# Step 1: Load components
print("\n[1/4] Loading components...")
from utils.pdf_loader import PDFLoader
from vector_store import get_vector_store
from config import DATASET_DIR

# Step 2: Select just 2 PDFs
print("\n[2/4] Processing 2 PDFs...")
loader = PDFLoader()

# Pick 2 smaller PDFs for faster testing
pdf1 = os.path.join(DATASET_DIR, "InfectiousDisease.pdf")  # 5.8 MB
pdf2 = os.path.join(DATASET_DIR, "General.pdf")  # 61 MB

print("Processing:", os.path.basename(pdf1))
chunks1 = loader.load_and_chunk_document(pdf1)
print("  Generated", len(chunks1), "chunks")

print("Processing:", os.path.basename(pdf2))
chunks2 = loader.load_and_chunk_document(pdf2)
print("  Generated", len(chunks2), "chunks")

all_chunks = chunks1 + chunks2
print("\nTotal chunks:", len(all_chunks))

# Step 3: Create vector database
print("\n[3/4] Building vector database...")
vs = get_vector_store()
vs.add_documents(all_chunks)
vs.save_index()

# Step 4: Verify
print("\n[4/4] Verifying...")
stats = vs.get_stats()
print("SUCCESS!")
print("  Stored:", stats['total_documents'], "chunks")
print("  Sources:", stats['sources'])

print("\n" + "="*70)
print("  DONE! 2 documents ingested successfully!")
print("="*70)
print("\nYou can now:")
print("  1. Go to http://localhost:8000/docs")
print("  2. Try the /query endpoint")
print("  3. Ask about infectious diseases or general medicine")
print("\n" + "="*70 + "\n")



