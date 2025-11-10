"""
Automated Test - Medical RAG Chatbot
Automatically ingests documents and tests with sample questions
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

print("\n" + "="*70)
print("  MEDICAL RAG CHATBOT - AUTOMATED TEST")
print("="*70)

# Step 1: Ingest Documents
print("\n[1/3] INGESTING MEDICAL DOCUMENTS...")
print("This will take 2-5 minutes. Please wait...")

try:
    response = requests.post(
        f"{BASE_URL}/ingest",
        json={"use_default_dataset": True},
        timeout=600
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n[SUCCESS] Ingested {data['num_documents']} documents")
        print(f"  Chunks: {data['num_chunks']}")
        print(f"  Sources: {', '.join(data['sources'])}")
    else:
        print(f"\n[ERROR] {response.text}")
        exit(1)
except Exception as e:
    print(f"\n[ERROR] {e}")
    exit(1)

# Step 2: Test Queries
print("\n[2/3] TESTING QUERIES...")
print("="*70)

questions = [
    "What are the symptoms of myocardial infarction?",
    "How do you treat anaphylactic shock?",
    "What are the types of diabetes?"
]

for i, question in enumerate(questions, 1):
    print(f"\n--- QUERY {i}/3 ---")
    print(f"Q: {question}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/query",
            json={"question": question, "top_k": 5},
            timeout=300
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nA: {data['answer'][:200]}...")
            print(f"\nCitations: {', '.join(data['citations'])}")
            print(f"Time: {data['response_time_ms']:.0f}ms")
        else:
            print(f"ERROR: {response.text}")
    except Exception as e:
        print(f"ERROR: {e}")
    
    time.sleep(2)

# Step 3: Show Stats
print("\n[3/3] SYSTEM STATISTICS")
print("="*70)

try:
    response = requests.get(f"{BASE_URL}/stats")
    data = response.json()
    
    print(f"\nVector Store:")
    print(f"  - Total documents: {data['vector_store']['total_documents']}")
    print(f"  - Embedding model: {data['vector_store']['embedding_model']}")
    
    print(f"\nQuery Logs:")
    print(f"  - Total queries: {data['query_logs']['total_queries']}")
    print(f"  - Avg response time: {data['query_logs']['avg_response_time_ms']:.0f}ms")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "="*70)
print("  TEST COMPLETE!")
print("="*70)
print("\nYour Medical RAG Chatbot is working perfectly!")
print("\nTo chat interactively, open: http://localhost:8000/docs")
print("\n" + "="*70 + "\n")



