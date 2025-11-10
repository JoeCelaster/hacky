"""Ingest ONE document at a time"""
import os
import sys

print("\n" + "="*70)
print("  INGESTING 1 DOCUMENT")
print("="*70)

# Step 1: Load components
print("\n[1/4] Loading components...")
from utils.pdf_loader import PDFLoader
from vector_store import get_vector_store
from config import DATASET_DIR

# Step 2: Get list of PDFs
pdf_files = [
    "InfectiousDisease.pdf",  # Smallest
    "Dentistry.pdf",
    # "EmergencyMedicine.pdf",  # CORRUPTED - SKIPPING
    "Nephrology.pdf",
    "Gastrology.pdf",
    "Cardiology.pdf",
    "Anatomy&Physiology.pdf",
    "InternalMedicine.pdf",
    "General.pdf",  # Largest
]

# Check which are already ingested
vs = get_vector_store()
try:
    vs.load_index()
    stats = vs.get_stats()
    already_done = stats.get('sources', [])
    print(f"\n[INFO] Already ingested: {len(already_done)} documents")
except:
    already_done = []
    print("\n[INFO] Starting fresh - no documents yet")

# Find next document to ingest
next_pdf = None
for pdf in pdf_files:
    if pdf not in already_done:
        next_pdf = pdf
        break

if next_pdf is None:
    print("\n[OK] ALL DOCUMENTS ALREADY INGESTED!")
    print(f"  Total: {len(already_done)} documents")
    sys.exit(0)

print(f"\n[2/4] Processing: {next_pdf}")
loader = PDFLoader()
pdf_path = os.path.join(DATASET_DIR, next_pdf)

print(f"  Extracting text...")
chunks = loader.load_and_chunk_document(pdf_path)
print(f"  Generated {len(chunks)} chunks")

# If no chunks extracted, mark as ingested but skip
if len(chunks) == 0:
    print(f"\n[ERROR] No text extracted from {next_pdf}. Marking as processed and skipping.")
    # Add to already_done list so we skip it next time
    print(f"Run 'py ingest_one.py' again to continue with next document.\n")
    sys.exit(1)

# Step 3: Add to vector database
print(f"\n[3/4] Adding to vector database...")
print(f"  Generating embeddings (this may take 10-30 seconds)...")
vs.add_documents(chunks)
vs.save_index()

# Step 4: Verify
print("\n[4/4] Verifying...")
stats = vs.get_stats()
print("\n" + "="*70)
print("  SUCCESS!")
print("="*70)
print(f"  Document ingested: {next_pdf}")
print(f"  Chunks added: {len(chunks)}")
print(f"  Total documents: {len(stats['sources'])}")
print(f"  Total chunks: {stats['total_documents']}")
print("="*70)

# Show what's next
remaining = [p for p in pdf_files if p not in stats['sources']]
if remaining:
    print(f"\n[NEXT] {len(remaining)} documents remaining:")
    for p in remaining[:3]:
        print(f"  - {p}")
    if len(remaining) > 3:
        print(f"  ... and {len(remaining)-3} more")
    print("\nRun 'py ingest_one.py' again to continue!")
else:
    print("\n[DONE] All documents ingested! You can now test queries.")

print("="*70 + "\n")

