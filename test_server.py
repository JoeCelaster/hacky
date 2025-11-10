"""Test if server is running"""
import time
import requests

print("Waiting for server to start...")
time.sleep(5)

try:
    response = requests.get('http://localhost:8000/health', timeout=5)
    print(f"\n✓ Server is running!")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except requests.exceptions.ConnectionError:
    print("\n✗ Server not responding yet.")
    print("Please run: py server.py")
except Exception as e:
    print(f"\n✗ Error: {e}")



