"""
Medical RAG Chatbot Server Launcher
Start the FastAPI application for the Medical RAG system
"""
import uvicorn
from config import API_HOST, API_PORT

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Medical RAG Chatbot API - Starting...")
    print("="*60)
    print("\n  Swagger UI: http://localhost:8000/docs")
    print("  ReDoc: http://localhost:8000/redoc\n")
    
    uvicorn.run(
        "app.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )


