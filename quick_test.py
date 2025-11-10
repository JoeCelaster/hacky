"""Quick test with a single query"""
import requests
import json

print("\n[TEST] Checking server and running ONE query...\n")

try:
    # Test health
    print("[1/2] Checking health...")
    r = requests.get("http://localhost:8000/health", timeout=10)
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        print(f"  Response: {r.json()}")
    
    # Test one simple query
    print("\n[2/2] Testing query...")
    print("  Question: What is diabetes?")
    print("  (This may take 1-3 minutes on first run - loading model...)")
    
    r = requests.post(
        "http://localhost:8000/query",
        json={"question": "What is diabetes?"},
        timeout=180  # 3 minutes
    )
    
    if r.status_code == 200:
        result = r.json()
        print(f"\n[SUCCESS] Got answer!")
        print(f"  Answer (first 300 chars): {result['answer'][:300]}...")
        print(f"  Citations: {len(result.get('citations', []))}")
        print(f"  Confidence: {result.get('confidence_score', 0):.2f}")
    else:
        print(f"\n[ERROR] Query failed: {r.status_code}")
        print(f"  Response: {r.text[:500]}")
        
except requests.exceptions.Timeout:
    print("[ERROR] Request timed out. Server may be loading model or crashed.")
except Exception as e:
    print(f"[ERROR] {e}")


