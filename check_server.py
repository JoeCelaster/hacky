"""Check if server is running and test endpoints"""
import time
import requests
import json

print("\nWaiting for server to start...")
time.sleep(8)

BASE_URL = "http://localhost:8000"

print("\n" + "="*70)
print("  TESTING MEDICAL RAG CHATBOT API")
print("="*70)

# Test 1: Health Check
print("\n[TEST 1] Health Check")
try:
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"  Status: {response.status_code}")
    data = response.json()
    print(f"  Vector Store Ready: {data.get('vector_store_ready')}")
    print(f"  Total Documents: {data.get('total_documents')}")
    print(f"  [OK] Server is running!")
except Exception as e:
    print(f"  [FAIL] {e}")
    exit(1)

# Test 2: Check Stats
print("\n[TEST 2] System Statistics")
try:
    response = requests.get(f"{BASE_URL}/stats", timeout=5)
    data = response.json()
    print(f"  Vector Store: {data['vector_store']['total_documents']} documents")
    print(f"  Query Logs: {data['query_logs']['total_queries']} queries")
    print(f"  [OK] Stats endpoint working")
except Exception as e:
    print(f"  [FAIL] {e}")

print("\n" + "="*70)
print("  SERVER IS RUNNING SUCCESSFULLY!")
print("="*70)
print("\n  Open in browser: http://localhost:8000/docs")
print("\n  Next steps:")
print("    1. Go to http://localhost:8000/docs")
print("    2. Try the /ingest endpoint to load medical PDFs")
print("    3. Try the /query endpoint to ask medical questions")
print("\n" + "="*70 + "\n")



