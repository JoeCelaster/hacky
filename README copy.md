# 🏥 Medical RAG Chatbot - Backend API

A complete **Retrieval-Augmented Generation (RAG)** system for medical Q&A that provides accurate, explainable, and citation-backed answers using **100% free and open-source** tools.

## 🌟 Features

- ✅ **No Paid APIs**: Completely free using Hugging Face models
- 🔍 **Semantic Search**: FAISS-based vector similarity search
- 📚 **Citation Support**: Every answer includes source citations
- 🧠 **Explainable AI**: Reasoning summaries for transparency
- 💾 **Query Logging**: SQLite database for tracking queries
- 📊 **Evaluation Metrics**: RAGAS-inspired evaluation system
- 🚀 **FastAPI Backend**: High-performance async API

## 📋 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | Python + FastAPI |
| **Orchestration** | LangChain |
| **Vector Store** | FAISS (local) |
| **Embeddings** | `BAAI/bge-base-en-v1.5` |
| **LLM** | `microsoft/BiomedGPT-LM-7B` (medical-focused) |
| **Fallback LLM** | `mistralai/Mistral-7B-Instruct-v0.3` |
| **Database** | SQLite |
| **PDF Processing** | PyPDF2 |

## 🏗️ Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP
       ▼
┌─────────────────────────────────────┐
│         FastAPI Server              │
│  (/query, /ingest, /evaluate)       │
└──────────┬──────────────────────────┘
           │
    ┌──────┴───────┐
    │              │
    ▼              ▼
┌────────┐    ┌──────────┐
│ Vector │    │   RAG    │
│ Store  │◄───┤ Pipeline │
│(FAISS) │    └────┬─────┘
└────────┘         │
                   ▼
          ┌────────────────┐
          │  LLM Generator │
          │  (BiomedGPT)   │
          └────────────────┘
```

## 📦 Installation

### 1. Navigate to Project

```bash
cd hack-a-cure
```

### 2. Run Installation Script

```bash
python install.py
```

This will:
- Check Python version
- Create necessary directories
- Install all dependencies
- Verify dataset

### 3. Alternative: Manual Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Download Models (Optional - will auto-download on first run)

The models will automatically download on first use, but you can pre-download them:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"
```

## 🚀 Quick Start

### 1. Start the Server

```bash
python server.py
```

Wait for the server to start, then you'll see:
```
Server starting...
Swagger UI: http://localhost:8000/docs
```

### 2. Ingest Medical Documents

The system comes with a dataset in `HackACure-Dataset/Dataset/`.

Use the `/ingest` endpoint:

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"use_default_dataset": true}'
```

### 3. Query the Chatbot

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the symptoms of myocardial infarction?",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "answer": "Myocardial infarction (heart attack) symptoms include...",
  "citations": ["Cardiology.pdf", "EmergencyMedicine.pdf"],
  "reasoning_summary": "Retrieved 5 relevant documents from: Cardiology.pdf (relevance: 0.89)...",
  "num_retrieved": 5,
  "response_time_ms": 1234.56
}
```

## 🔌 API Endpoints

### General

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check and status |
| `/stats` | GET | System statistics |

### Data Ingestion

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Ingest documents from directory |
| `/upload` | POST | Upload and ingest files |
| `/reset` | DELETE | Clear vector store |

### Query & Evaluation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Ask a medical question |
| `/evaluate` | POST | Evaluate response quality |

## 📊 API Examples

### Health Check

```bash
curl http://localhost:8000/health
```

### Upload Custom Documents

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@medical_paper.pdf" \
  -F "files=@clinical_notes.txt"
```

### Evaluate Query Quality

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is diabetes mellitus?",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "query": "What is diabetes mellitus?",
  "faithfulness_score": 0.85,
  "context_recall": 0.78,
  "context_precision": 0.82,
  "answer_relevancy": 0.88
}
```

### Get Statistics

```bash
curl http://localhost:8000/stats
```

## 📁 Project Structure

```
hack-a-cure/
├── app/
│   ├── __init__.py
│   └── main.py                 # FastAPI application
├── models/
│   ├── __init__.py
│   └── query_log.py            # SQLite query logging
├── utils/
│   ├── __init__.py
│   ├── pdf_loader.py           # PDF text extraction
│   └── text_splitter.py        # Advanced text chunking
├── HackACure-Dataset/
│   └── Dataset/                # Medical PDFs (9 domains)
│       ├── Cardiology.pdf
│       ├── EmergencyMedicine.pdf
│       └── ...
├── data/                       # Auto-generated data directory
│   ├── faiss_index/            # Vector store
│   ├── uploads/                # Uploaded files
│   └── query_logs.db           # Query logs
├── config.py                   # Configuration settings
├── vector_store.py             # FAISS vector store management
├── rag_pipeline.py             # Core RAG logic
├── server.py                   # Server launcher
├── install.py                  # Installation script
├── test_system.py              # System test suite
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker orchestration
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Models
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
GENERATOR_MODEL = "microsoft/BiomedGPT-LM-7B"

# Chunking
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50

# Retrieval
TOP_K_RETRIEVAL = 5
SIMILARITY_THRESHOLD = 0.5

# Generation
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
```

## 🧪 Testing the System

### 1. Run the Server

```bash
python server.py
```

Or with custom settings:

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Access Interactive Docs

Open your browser to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3. Sample Medical Questions

Try these queries:

```bash
# Cardiology
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the risk factors for heart disease?"}'

# Emergency Medicine
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "How do you treat anaphylactic shock?"}'

# Internal Medicine
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the types of diabetes?"}'
```

## 🎯 RAG Pipeline Flow

1. **Ingestion Phase**
   - Load PDFs/TXT files
   - Split into ~500 token chunks
   - Generate embeddings using `all-MiniLM-L6-v2`
   - Store in FAISS index

2. **Query Phase**
   - Embed user question
   - Retrieve top-k similar chunks from FAISS
   - Filter by similarity threshold
   - Format context with citations

3. **Generation Phase**
   - Construct prompt with context
   - Generate answer using BiomedGPT
   - Extract citations from sources
   - Create reasoning summary

4. **Logging Phase**
   - Store query, answer, citations
   - Calculate response metrics
   - Save to SQLite database

## 🔬 Evaluation Metrics

The system provides RAGAS-inspired metrics:

- **Faithfulness**: Does the answer align with retrieved context?
- **Context Recall**: How much of the query is covered by context?
- **Context Precision**: How relevant are the retrieved documents?
- **Answer Relevancy**: Is the answer substantial and relevant?

## 🚨 Troubleshooting

### Model Download Issues

If models fail to download:

```bash
# Set Hugging Face cache directory
export HF_HOME=/path/to/large/disk

# Or use fallback model
# In config.py, the system will auto-fallback to Mistral if BiomedGPT fails
```

### Out of Memory

For low-memory systems:

```python
# In rag_pipeline.py, reduce batch size
embeddings = self.embedding_model.encode(
    texts,
    batch_size=8  # Reduce from 32
)

# Or use CPU-only mode
device = "cpu"  # Force CPU in rag_pipeline.py
```

### FAISS Index Errors

```bash
# Clear and rebuild index
curl -X DELETE "http://localhost:8000/reset"
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"use_default_dataset": true}'
```

## 🔐 Security Notes

- This is a development server. For production:
  - Add authentication (JWT, OAuth)
  - Use HTTPS
  - Add rate limiting
  - Sanitize file uploads
  - Use environment variables for secrets

## 📈 Performance Tips

1. **Use GPU**: Install `faiss-gpu` and `torch` with CUDA support
2. **Batch Processing**: Process multiple queries together
3. **Index Optimization**: For large datasets, use `IndexIVFFlat`
4. **Model Quantization**: Use 4-bit/8-bit quantization for LLM
5. **Caching**: Cache frequent queries

## 🤝 Contributing

This is a hackathon project for medical Q&A. Contributions welcome!

## 📄 License

MIT License - Feel free to use for educational and research purposes.

## 🙏 Acknowledgments

- **Hugging Face**: For free model hosting
- **FAISS**: For efficient vector search
- **LangChain**: For RAG orchestration
- **FastAPI**: For the excellent web framework

## 🧪 Testing

Run the automated test suite:

```bash
python test_system.py
```

This will test all endpoints and verify the system is working correctly.

## 📞 Support & Troubleshooting

For issues, check:
- Server logs: Look for errors in terminal
- `/health` endpoint: Check system status
- `/stats` endpoint: Monitor usage
- Interactive docs: http://localhost:8000/docs

## 📁 Quick Commands

```bash
# Install & setup
python install.py

# Start server
python server.py

# Test system
python test_system.py

# Access API docs
# Open browser: http://localhost:8000/docs
```

---

**Built with ❤️ for HackACure Hackathon**

🌟 **Free. Open Source. No API Keys Required.** 🌟



