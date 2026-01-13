# Hacky – AI Medical Assistant 🩺

Hacky is an AI-powered medical assistance platform built to provide intelligent, context-aware answers from medical documents using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG). It is designed for healthcare research, medical education, and experimental clinical decision support systems.

Hacky combines LLMs with vector databases to create a smart medical assistant that can understand medical questions in natural language, search through medical knowledge, retrieve the most relevant context, and generate accurate and meaningful responses.

This project is ideal for building:
- Medical chatbots  
- Clinical research assistants  
- Medical knowledge retrieval systems  
- Domain-specific healthcare AI tools  

---

## 🧠 Core Technologies

- Python for backend and AI pipelines  
- Large Language Models (LLMs) for reasoning and generation  
- Vector Databases (Chroma / FAISS / Milvus) for semantic search  
- Embedding models for converting medical text into vectors  
- Retrieval-Augmented Generation (RAG) architecture  
- REST APIs for AI services  
- Docker & Docker Compose for containerization  
- Secure environment variable handling  

---

## ✨ Features

- 🩻 Medical document ingestion  
- 🔎 Semantic search on medical data  
- 💬 AI chat with medical context  
- 📚 Knowledge-based question answering  
- ⚡ Fast and scalable architecture  
- 🔐 Secure API key handling  
- 🐳 Docker-ready deployment  
- 🧪 Test scripts for pipeline validation  

---

## ⚙️ Setup

Clone the repository:
```bash
git clone https://github.com/JoeCelaster/hacky.git
cd hacky
```

Create a virtual environment and install dependencies:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set environment variables:
```
OPENAI_API_KEY=your_api_key_here

```
Run the server:
```
python server.py
```

Ingest medical documents:
```
python ingest_now.py --source path/to/medical/data
```

Chat with the AI:
```
python chat_with_bot.py

```
Run tests:
```
pytest

```
Run with Docker:
```
docker build -t hacky .
docker-compose up
```
