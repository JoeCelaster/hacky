"""Wait for server and run ingestion"""
import time
import requests

print("Waiting for server to be ready...")
time.sleep(10)

# Check if server is responding
for i in range(5):
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("[OK] Server is ready!")
            break
    except:
        print(f"Attempt {i+1}/5 - waiting...")
        time.sleep(3)

# Now run ingestion
print("\nStarting ingestion...")
exec(open("ingest_now.py").read())



