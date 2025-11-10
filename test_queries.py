"""Test the RAG system with various medical queries"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

# Test queries covering different medical topics
test_queries = [
    {
        "question": "What are the symptoms of diabetes?",
        "category": "General Medicine"
    },
    {
        "question": "How is hypertension treated?",
        "category": "Cardiology"
    },
    {
        "question": "What are the stages of chronic kidney disease?",
        "category": "Nephrology"
    },
    {
        "question": "What antibiotics are used for pneumonia?",
        "category": "Infectious Disease"
    },
    {
        "question": "What are the causes of gastritis?",
        "category": "Gastrology"
    },
    {
        "question": "How do you perform dental extraction?",
        "category": "Dentistry"
    },
    {
        "question": "What is the structure of the human heart?",
        "category": "Anatomy"
    },
]

print("\n" + "="*80)
print("  MEDICAL RAG SYSTEM - COMPREHENSIVE TEST")
print("="*80)

# Check server health
print("\n[1/3] Checking server health...")
try:
    response = requests.get(f"{BASE_URL}/health", timeout=30)
    health = response.json()
    print(f"  Server Status: {health.get('status', 'unknown')}")
    if 'model' in health:
        print(f"  Model: {health['model']}")
    print("  [OK] Server is running!")
except Exception as e:
    print(f"  [ERROR] Server not responding: {e}")
    print("\n  Please make sure the server is running:")
    print("    py server.py")
    exit(1)

# Check stats
print("\n[2/3] Checking ingested documents...")
try:
    response = requests.get(f"{BASE_URL}/stats", timeout=30)
    stats = response.json()
    print(f"  Total Documents: {stats['total_documents']}")
    print(f"  Total Queries: {stats['total_queries']}")
    print(f"  Sources: {', '.join(stats['sources'][:3])}...")
except Exception as e:
    print(f"  [ERROR] Could not fetch stats: {e}")

# Test queries
print("\n[3/3] Testing medical queries...")
print("="*80)

successful = 0
failed = 0

for i, test in enumerate(test_queries, 1):
    print(f"\n[TEST {i}/{len(test_queries)}] {test['category']}")
    print(f"Q: {test['question']}")
    print("-" * 80)
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/query",
            json={"question": test['question']},
            timeout=120  # 2 minutes for first query (model loading)
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            # Display answer
            print(f"A: {result['answer'][:200]}...")  # First 200 chars
            
            # Display citations
            print(f"\nCitations: {len(result.get('citations', []))} sources")
            for j, citation in enumerate(result.get('citations', [])[:2], 1):
                print(f"  [{j}] {citation.get('source', 'Unknown')} (page {citation.get('page', '?')})")
            
            # Display metrics
            print(f"\nMetrics:")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Confidence: {result.get('confidence_score', 0):.2f}")
            print(f"  Relevant Chunks: {len(result.get('retrieved_context', []))}")
            
            successful += 1
            print("[OK] Query successful!")
            
        else:
            print(f"[ERROR] Query failed: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            failed += 1
            
    except requests.exceptions.Timeout:
        print("[ERROR] Query timed out (this may happen on first query while loading model)")
        print("        Please run the test again - subsequent queries will be faster.")
        failed += 1
        break
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")
        failed += 1
    
    # Small delay between queries
    if i < len(test_queries):
        time.sleep(2)

# Summary
print("\n" + "="*80)
print("  TEST SUMMARY")
print("="*80)
print(f"  Total Tests: {len(test_queries)}")
print(f"  Successful: {successful}")
print(f"  Failed: {failed}")
print(f"  Success Rate: {(successful/len(test_queries)*100):.1f}%")
print("="*80)

if successful == len(test_queries):
    print("\n[SUCCESS] All tests passed! RAG system is working perfectly!")
elif successful > 0:
    print("\n[PARTIAL] Some tests passed. System is functional but may need optimization.")
else:
    print("\n[FAILED] No tests passed. Please check the logs and model loading.")

print("\n")

