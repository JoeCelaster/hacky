# 🚀 Medical RAG Chatbot - Quick Start Guide

## 📋 Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (16GB recommended)
- 20GB+ disk space (for models and data)
- Internet connection (for first-time model download)

## ⚡ 3-Step Quick Start

### Step 1: Install

```bash
python install.py
```

This will install all dependencies and set up the environment.

### Step 2: Start Server

```bash
python server.py
```

Wait for the message: `Application startup complete`

### Step 3: Test System

Open a new terminal and run:

```bash
python test_system.py
```

## 🌐 Access the API

Once the server is running:

- **Interactive API Docs**: http://localhost:8000/docs
- **API Base URL**: http://localhost:8000
- **Alternative Docs**: http://localhost:8000/redoc

## 📚 First Query

### Using Browser (Swagger UI)

1. Go to http://localhost:8000/docs
2. Click on `/ingest` endpoint
3. Click "Try it out"
4. Set `use_default_dataset` to `true`
5. Click "Execute"
6. Wait for ingestion to complete
7. Click on `/query` endpoint
8. Enter your medical question
9. Click "Execute"

### Using Command Line

```bash
# 1. Ingest medical documents
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"use_default_dataset": true}'

# 2. Ask a medical question
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the symptoms of heart attack?",
    "top_k": 5
  }'
```

## 📁 Project Structure

```
hack-a-cure/
├── server.py           # Start the API server
├── install.py          # Installation script
├── test_system.py      # Test all endpoints
├── config.py           # Configuration
├── vector_store.py     # FAISS vector database
├── rag_pipeline.py     # RAG logic
├── requirements.txt    # Dependencies
├── app/                # FastAPI application
├── models/             # Database models
├── utils/              # Utilities (PDF loader, etc.)
└── HackACure-Dataset/  # Medical PDFs (9 domains)
```

## 🎯 Key Features

✅ **8 API Endpoints**
- `/health` - Check system status
- `/ingest` - Add medical documents
- `/upload` - Upload custom files
- `/query` - Ask medical questions
- `/evaluate` - Evaluate response quality
- `/stats` - System statistics
- `/reset` - Clear database

✅ **Citation Support**
- Every answer includes source citations
- Reasoning summaries explain source selection

✅ **Evaluation Metrics**
- Faithfulness score
- Context recall
- Context precision
- Answer relevancy

✅ **Medical Domains Covered**
1. Anatomy & Physiology
2. Cardiology
3. Dentistry
4. Emergency Medicine
5. Gastrology
6. General Medicine
7. Infectious Disease
8. Internal Medicine
9. Nephrology

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Embedding model
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Generator model
GENERATOR_MODEL = "microsoft/BiomedGPT-LM-7B"

# Retrieval settings
TOP_K_RETRIEVAL = 5
CHUNK_SIZE = 500

# Generation settings
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
```

## 🐛 Troubleshooting

### Server won't start
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000  # Windows
lsof -i :8000                  # Linux/Mac
```

### Models downloading slowly
First run downloads ~2-7GB of models. This is normal. Ensure good internet connection.

### Out of memory
Use CPU-only mode by editing `rag_pipeline.py`:
```python
device = "cpu"  # Force CPU
```

### No documents found
Verify dataset exists:
```bash
ls HackACure-Dataset/Dataset/
```

## 📞 Get Help

### Check System Status
```bash
curl http://localhost:8000/health
```

### View System Statistics
```bash
curl http://localhost:8000/stats
```

### Test All Endpoints
```bash
python test_system.py
```

## 🎓 Example Medical Queries

Try these questions after ingesting documents:

**Cardiology:**
```
What are the risk factors for coronary artery disease?
How is acute myocardial infarction diagnosed?
```

**Emergency Medicine:**
```
How do you manage anaphylactic shock?
What is the protocol for stroke management?
```

**Gastrology:**
```
What are the symptoms of GERD?
How is peptic ulcer disease treated?
```

**Nephrology:**
```
What are the stages of chronic kidney disease?
How is acute kidney injury managed?
```

**Infectious Disease:**
```
What is the treatment for bacterial meningitis?
How is sepsis diagnosed and treated?
```

## 🚀 Advanced Usage

### Upload Custom Documents
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@medical_paper.pdf" \
  -F "files=@clinical_notes.txt"
```

### Evaluate Response Quality
```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is diabetes?"}'
```

### Reset and Start Fresh
```bash
curl -X DELETE "http://localhost:8000/reset"
```

## 🐳 Docker Deployment

```bash
# Build and run with Docker
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## 📊 Performance Tips

1. **GPU Support**: Install `torch` with CUDA for faster processing
2. **Batch Queries**: Process multiple queries together
3. **Adjust Chunk Size**: Larger chunks = more context, slower retrieval
4. **Lower Top-K**: Fewer retrieved docs = faster responses

## 🎯 Common Commands Cheat Sheet

```bash
# Setup
python install.py

# Start server
python server.py

# Test system
python test_system.py

# Check health
curl http://localhost:8000/health

# Ingest documents
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"use_default_dataset": true}'

# Query chatbot
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here"}'

# View stats
curl http://localhost:8000/stats
```

## 📖 Further Reading

- **README.md** - Complete documentation
- **API Documentation** - http://localhost:8000/docs (when server is running)
- **Configuration** - See `config.py` for all settings

---

**Ready to start? Run: `python install.py`** 🚀

**Need help? Check server logs or visit: http://localhost:8000/docs**




