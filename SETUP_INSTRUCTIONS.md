# 🚀 Medical RAG Chatbot - Setup Instructions

## ✅ Complete Build Summary

Your Medical RAG Chatbot is now ready! Here's what has been built:

### 📦 Project Structure
```
hack-a-cure/
├── Core Application
│   ├── server.py              # API server launcher
│   ├── config.py              # Configuration (BGE embeddings + BioMistral-7B-GPTQ)
│   ├── rag_pipeline.py        # RAG logic with GPTQ support
│   ├── vector_store.py        # FAISS vector database
│   ├── app/main.py            # FastAPI endpoints
│   ├── models/query_log.py    # SQLite logging
│   └── utils/pdf_loader.py    # PDF processing
│
├── Installation & Testing
│   ├── quick_install.py       # Fast installation script
│   ├── test_quick.py          # Quick test suite
│   ├── test_system.py         # Full system tests
│   ├── install.py             # Interactive installer
│   └── run_all.bat            # Windows batch script
│
├── Documentation
│   ├── README.md              # Complete documentation
│   ├── QUICKSTART.md          # Quick start guide
│   └── SETUP_INSTRUCTIONS.md  # This file
│
├── Medical Dataset
│   └── HackACure-Dataset/Dataset/
│       ├── Anatomy&Physiology.pdf
│       ├── Cardiology.pdf
│       ├── Dentistry.pdf
│       ├── EmergencyMedicine.pdf
│       ├── Gastrology.pdf
│       ├── General.pdf
│       ├── InfectiousDisease.pdf
│       ├── InternalMedicine.pdf
│       └── Nephrology.pdf
│
└── Configuration
    ├── requirements.txt       # Python dependencies
    ├── Dockerfile            # Docker deployment
    └── docker-compose.yml    # Docker orchestration
```

## 🎯 Configured Models (Optimized for RTX 2050)

### Embedding Model
- **Model**: `BAAI/bge-base-en-v1.5`
- **Size**: 420 MB
- **Dimensions**: 768
- **Purpose**: Convert text to vectors for similarity search

### Generator Model  
- **Model**: `TheBloke/BioMistral-7B-GPTQ`
- **Size**: ~3.5 GB (4-bit quantized)
- **VRAM**: Fits in 4GB GPU
- **Specialization**: Medical domain (trained on PubMed, clinical texts)
- **Fallback**: `microsoft/phi-2` (2.7B model)

## 🚀 Installation & Running

### Option 1: Automatic Installation (Recommended)

```bash
# Run the complete installation script
py quick_install.py

# Test the installation
py test_quick.py

# Start the server
py server.py
```

### Option 2: Manual Installation

```bash
# Install dependencies
py -m pip install fastapi uvicorn torch transformers sentence-transformers
py -m pip install faiss-cpu langchain langchain-community PyPDF2
py -m pip install numpy python-multipart aiofiles

# Optional: GPTQ for GPU optimization
py -m pip install auto-gptq optimum accelerate

# Start server
py server.py
```

### Option 3: One-Click (Windows)

```bash
# Double-click or run:
run_all.bat
```

## 📊 API Endpoints

Once the server is running at `http://localhost:8000`:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/docs` | GET | Interactive API documentation |
| `/health` | GET | Check system status |
| `/ingest` | POST | Load medical documents into vector DB |
| `/upload` | POST | Upload custom PDF/TXT files |
| `/query` | POST | Ask medical questions |
| `/evaluate` | POST | Evaluate response quality |
| `/stats` | GET | System statistics |
| `/reset` | DELETE | Clear vector database |

## 🧪 Testing the System

### Step 1: Start the Server

```bash
py server.py
```

Wait for:
```
Server starting...
Swagger UI: http://localhost:8000/docs
```

### Step 2: Ingest Medical Documents

Open a new terminal:

```bash
curl -X POST "http://localhost:8000/ingest" -H "Content-Type: application/json" -d "{\"use_default_dataset\": true}"
```

Or use the Swagger UI at `http://localhost:8000/docs` and click on `/ingest` endpoint.

### Step 3: Query the System

```bash
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d "{\"question\": \"What are the symptoms of heart attack?\", \"top_k\": 5}"
```

Expected response:
```json
{
  "answer": "Myocardial infarction (heart attack) symptoms include...",
  "citations": ["Cardiology.pdf", "EmergencyMedicine.pdf"],
  "reasoning_summary": "Retrieved 5 relevant documents...",
  "num_retrieved": 5,
  "response_time_ms": 2341.5
}
```

## 🎮 GPU Support (RTX 2050)

Your system is configured for RTX 2050 (4GB VRAM):

### Automatic Detection
The system will automatically:
1. Detect CUDA availability
2. Load models with optimal settings
3. Use GPTQ quantization if available
4. Fall back to CPU if GPU unavailable

### Manual GPU Check

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Expected VRAM Usage
- **Embedding Model** (CPU): ~0.5GB RAM
- **BioMistral-7B-GPTQ** (GPU): ~3.5GB VRAM
- **Total GPU Usage**: ~3.5GB (leaves 0.5GB buffer)

## 📈 Performance Expectations

### First-Time Setup
- **Model Download**: 3-4 GB (~10-30 min depending on internet)
- **Document Ingestion**: 2-5 minutes (9 PDFs, ~1000 chunks)
- **First Query**: 30-60 seconds (model loading)

### After Setup
- **Query Response**: 2-5 seconds
- **Token Generation**: 30-50 tokens/second
- **Concurrent Users**: 5-10 (depending on GPU memory)

## 🔧 Troubleshooting

### Issue 1: Python not found
**Solution**: Use `py` instead of `python`:
```bash
py server.py
```

### Issue 2: GPTQ installation fails
**Solution**: This is optional. The system will work with standard models:
```bash
# Skip GPTQ and use fallback model
# Edit config.py:
GENERATOR_MODEL = "microsoft/phi-2"
```

### Issue 3: Out of memory on GPU
**Solution**: The code will automatically fall back to Phi-2 (smaller model)

### Issue 4: Models downloading slowly
**Solution**: Be patient. First run downloads ~4GB of models. This is normal.

### Issue 5: Port 8000 already in use
**Solution**: Kill the process or use different port:
```bash
py -m uvicorn app.main:app --port 8001
```

## 📝 Usage Examples

### Example 1: Cardiology Query
```json
{
  "question": "What are the risk factors for coronary artery disease?",
  "top_k": 5
}
```

### Example 2: Emergency Medicine
```json
{
  "question": "How do you manage anaphylactic shock?",
  "top_k": 5
}
```

### Example 3: Gastroenterology
```json
{
  "question": "What are the symptoms of gastroesophageal reflux disease?",
  "top_k": 5
}
```

## 🎯 Complete Workflow

```bash
# 1. Install (first time only)
py quick_install.py

# 2. Test installation
py test_quick.py

# 3. Start server
py server.py

# 4. In another terminal: Ingest documents
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d "{\"use_default_dataset\": true}"

# 5. Query the system
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"question\": \"What is diabetes?\", \"top_k\": 5}"

# 6. Check statistics
curl http://localhost:8000/stats

# 7. Access interactive docs
# Open browser: http://localhost:8000/docs
```

## 🎉 You're Ready!

Your Medical RAG Chatbot is fully configured and ready to use:

✅ **Backend**: FastAPI with 8 REST endpoints  
✅ **RAG Pipeline**: Retrieval + Generation + Citations  
✅ **Vector Store**: FAISS with BGE embeddings  
✅ **LLM**: BioMistral-7B-GPTQ (medical-specialized)  
✅ **Dataset**: 9 medical domains  
✅ **GPU Support**: Optimized for RTX 2050  
✅ **Logging**: SQLite query tracking  
✅ **Evaluation**: RAGAS-inspired metrics  

## 📞 Quick Commands Reference

```bash
# Install
py quick_install.py

# Test
py test_quick.py

# Run server
py server.py

# Test system (when server is running)
py test_system.py

# Access docs
http://localhost:8000/docs
```

## 🌟 Key Features

1. **Citation Support**: Every answer includes source PDFs
2. **Explainable AI**: Reasoning summaries for transparency
3. **Medical Specialization**: BioMistral model trained on medical literature
4. **GPU Optimization**: 4-bit quantization for 4GB VRAM
5. **Query Logging**: Track all queries and responses
6. **Evaluation Metrics**: Faithfulness, recall, precision, relevancy
7. **Multi-Domain**: 9 medical specialties covered
8. **Free & Open Source**: No API keys required

---

**Need Help?** Check the logs in the terminal where you ran `py server.py`

**Built for HackACure Hackathon** 🏥💉



