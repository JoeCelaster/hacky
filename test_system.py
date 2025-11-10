"""
Medical RAG Chatbot - System Testing Script
Tests all API endpoints and functionality
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("\n🔍 Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()


def test_ingest():
    """Test ingestion of default dataset"""
    print("\n📚 Testing /ingest endpoint...")
    response = requests.post(
        f"{BASE_URL}/ingest",
        json={"use_default_dataset": True}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()


def test_query(question: str):
    """Test query endpoint"""
    print(f"\n❓ Testing /query endpoint with: '{question}'")
    response = requests.post(
        f"{BASE_URL}/query",
        json={
            "question": question,
            "top_k": 5
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Answer: {result['answer'][:200]}...")
        print(f"\n📖 Citations: {', '.join(result['citations'])}")
        print(f"\n🧠 Reasoning: {result['reasoning_summary']}")
        print(f"\n⏱️ Response time: {result['response_time_ms']:.2f}ms")
    else:
        print(f"Error: {response.text}")
    return response.json() if response.status_code == 200 else None


def test_evaluate(question: str):
    """Test evaluation endpoint"""
    print(f"\n📊 Testing /evaluate endpoint with: '{question}'")
    response = requests.post(
        f"{BASE_URL}/evaluate",
        json={
            "question": question,
            "top_k": 5
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\nMetrics:")
        print(f"  - Faithfulness: {result['faithfulness_score']:.2f}")
        print(f"  - Context Recall: {result['context_recall']:.2f}")
        print(f"  - Context Precision: {result['context_precision']:.2f}")
        print(f"  - Answer Relevancy: {result['answer_relevancy']:.2f}")
    else:
        print(f"Error: {response.text}")
    return response.json() if response.status_code == 200 else None


def test_stats():
    """Test stats endpoint"""
    print("\n📈 Testing /stats endpoint...")
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()


def main():
    """Run all tests"""
    print("=" * 60)
    print("Medical RAG Chatbot - System Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Health check
        health = test_health()
        
        # Test 2: Ingest documents (if not already done)
        if not health.get('vector_store_ready', False):
            print("\n⚠️ Vector store not ready. Ingesting documents...")
            test_ingest()
            time.sleep(2)
        else:
            print("\n✅ Vector store already ready!")
        
        # Test 3: Query examples
        test_questions = [
            "What are the symptoms of myocardial infarction?",
            "How do you treat anaphylactic shock?",
            "What are the types of diabetes?"
        ]
        
        for question in test_questions:
            test_query(question)
            time.sleep(1)
        
        # Test 4: Evaluate a query
        test_evaluate(test_questions[0])
        
        # Test 5: Get statistics
        test_stats()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to API server.")
        print("Make sure the server is running:")
        print("  python server.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()




