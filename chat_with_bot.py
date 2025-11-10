"""
Simple Chat Interface for Medical RAG Chatbot
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def check_server():
    """Check if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def ingest_documents():
    """Ingest medical documents"""
    print("\n" + "="*70)
    print("  STEP 1: INGESTING MEDICAL DOCUMENTS")
    print("="*70)
    print("\nThis will load 9 medical PDFs into the system...")
    print("Processing: Anatomy, Cardiology, Dentistry, Emergency Medicine,")
    print("           Gastrology, General, Infectious Disease,")
    print("           Internal Medicine, Nephrology")
    print("\nThis may take 2-5 minutes. Please wait...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/ingest",
            json={"use_default_dataset": True},
            timeout=600  # 10 minutes timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n[SUCCESS] Ingestion Complete!")
            print(f"  - Documents: {data['num_documents']}")
            print(f"  - Chunks: {data['num_chunks']}")
            print(f"  - Sources: {', '.join(data['sources'][:3])}...")
            return True
        else:
            print(f"\n[ERROR] Failed: {response.text}")
            return False
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return False

def query_chatbot(question):
    """Ask a question to the chatbot"""
    print(f"\n[QUERY] {question}")
    print("Generating answer...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/query",
            json={"question": question, "top_k": 5},
            timeout=300  # 5 minutes for first query (model download)
        )
        
        if response.status_code == 200:
            data = response.json()
            print("\n" + "="*70)
            print("  ANSWER")
            print("="*70)
            print(f"\n{data['answer']}\n")
            print("-"*70)
            print(f"Citations: {', '.join(data['citations'])}")
            print(f"Response Time: {data['response_time_ms']:.0f}ms")
            print("="*70)
            return True
        else:
            print(f"\n[ERROR] {response.text}")
            return False
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return False

def main():
    print("\n" + "="*70)
    print("  MEDICAL RAG CHATBOT - CHAT INTERFACE")
    print("="*70)
    
    # Check if server is running
    if not check_server():
        print("\n[ERROR] Server not running!")
        print("Please start the server first: py server.py")
        return
    
    print("\n[OK] Server is running at http://localhost:8000")
    
    # Check if documents are ingested
    response = requests.get(f"{BASE_URL}/health")
    health = response.json()
    
    if not health['vector_store_ready']:
        print("\n[INFO] Documents not yet ingested.")
        choice = input("\nDo you want to ingest medical documents now? (y/n): ")
        if choice.lower() == 'y':
            if not ingest_documents():
                return
        else:
            print("\n[INFO] Please ingest documents first using:")
            print("  - Browser: http://localhost:8000/docs -> /ingest endpoint")
            print("  - Or run this script again and choose 'y'")
            return
    else:
        print(f"\n[OK] Vector store ready with {health['total_documents']} documents")
    
    # Ask questions
    print("\n" + "="*70)
    print("  READY TO CHAT!")
    print("="*70)
    print("\nYou can ask medical questions. Type 'quit' to exit.")
    print("\nExample questions:")
    print("  - What are the symptoms of heart attack?")
    print("  - How do you treat anaphylactic shock?")
    print("  - What are the types of diabetes?")
    print("  - What is the treatment for bacterial meningitis?")
    
    # Sample questions for quick testing
    sample_questions = [
        "What are the symptoms of myocardial infarction?",
        "How do you manage anaphylactic shock?",
        "What are the risk factors for coronary artery disease?"
    ]
    
    print("\n" + "="*70)
    print("  RUNNING SAMPLE QUERIES")
    print("="*70)
    
    for question in sample_questions:
        if query_chatbot(question):
            time.sleep(2)  # Wait a bit between queries
    
    # Interactive mode
    print("\n" + "="*70)
    print("  INTERACTIVE MODE")
    print("="*70)
    print("\nNow you can ask your own questions!")
    print("(Note: First query may take 1-2 minutes to load the model)\n")
    
    while True:
        try:
            question = input("\nYour question (or 'quit'): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using Medical RAG Chatbot!")
                break
            
            if not question:
                continue
            
            query_chatbot(question)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break

if __name__ == "__main__":
    main()



