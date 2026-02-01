# 📄 AI-Powered Contract Processing & Analysis System

A complete end-to-end system for generating, processing, extracting, and querying contract data using state-of-the-art open-source LLMs and AI technologies.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![AI Powered](https://img.shields.io/badge/AI-Powered-purple.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)
- [Security](#security)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This system provides a complete pipeline for contract lifecycle management:

1. **🔨 Contract Generation**: Generate realistic contract PDFs (MSA, SOW, NDA, etc.) using LLMs
2. **📖 Text Extraction**: Extract text from PDFs using multi-modal vision models, OCR, and direct extraction
3. **🔍 Entity Extraction**: Extract 48+ contract attributes using RAG (FAISS) + LLM
4. **💬 Natural Language Querying**: Query contracts using natural language powered by LLM-generated pandas code

**Perfect for:**
- Legal tech companies
- Contract management systems
- Due diligence automation
- Contract analytics platforms
- Legal research and training

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Contract Processing Pipeline                 │
└─────────────────────────────────────────────────────────────────┘

Step 1: PDF Generation
┌──────────────────────────────┐
│  contract_generator.py       │
│  • Flan-T5 LLM              │
│  • ReportLab PDF generation  │
│  • 100 contracts (30+ pages) │
└──────────────┬───────────────┘
               │
               ▼
        [PDF Files] ──────────────────────┐
               │                          │
               ▼                          │
Step 2: Text Extraction                  │
┌──────────────────────────────┐         │
│  extract_text.py             │         │
│  • BLIP-2 Vision Model       │         │
│  • Tesseract OCR (fallback)  │         │
│  • PyPDF2 (fallback)         │         │
└──────────────┬───────────────┘         │
               │                          │
               ▼                          │
        [Text Files]                      │
               │                          │
               ▼                          │
Step 3: Entity Extraction                │
┌──────────────────────────────┐         │
│  extract_attributes.py       │         │
│  • FAISS Vector Search       │         │
│  • Flan-T5 LLM              │         │
│  • 48 entity attributes      │         │
└──────────────┬───────────────┘         │
               │                          │
               ▼                          │
    [contract_entities.csv] ──────────┐  │
               │                       │  │
               │                       │  │
Step 4: Query System                  │  │
┌──────────────────────────────┐      │  │
│  Flask API (query_engine.py) │      │  │
│  • CodeGen LLM               │◄─────┘  │
│  • Safe pandas execution     │         │
│  • REST API endpoints        │         │
└──────────────┬───────────────┘         │
               │                          │
               ▼                          │
┌──────────────────────────────┐         │
│  Streamlit Web App           │         │
│  • Natural language queries  │         │
│  • Interactive visualizations│         │
│  • Entity browser            │◄────────┘
└──────────────────────────────┘
```

---

## ✨ Features

### 1. Contract PDF Generation
- ✅ Generate 100 realistic multi-page contracts (MSA, SOW, NDA, etc.)
- ✅ 30+ pages per contract with 300+ words per page
- ✅ AI-generated legal language using Flan-T5
- ✅ Realistic company names, addresses, dates
- ✅ Professional formatting with ReportLab

### 2. Multi-Tier Text Extraction
- ✅ **Tier 1**: Multi-modal vision model (BLIP-2) for accurate text extraction
- ✅ **Tier 2**: OCR (Tesseract) fallback
- ✅ **Tier 3**: Direct PDF text extraction (PyPDF2) fallback
- ✅ Page-by-page processing with base64 image conversion
- ✅ Automatic method selection based on success

### 3. RAG-Based Entity Extraction
- ✅ **48 contract attributes** extracted automatically:
  - Party information (names, addresses, contacts, signatories)
  - Contract basics (type, dates, term, renewal)
  - Financial terms (value, payment schedule, currency)
  - Scope & deliverables
  - Legal terms (governing law, liability, dispute resolution)
  - IP ownership & confidentiality
  - Termination & warranties
- ✅ **FAISS vector search** for semantic chunk retrieval
- ✅ **LLM-based extraction** from relevant chunks
- ✅ **CSV output** with all attributes

### 4. Natural Language Query System
- ✅ **Natural language queries** - ask questions in plain English
- ✅ **LLM code generation** - CodeGen automatically writes pandas code
- ✅ **Safe execution** - restricted environment (no file I/O, network, or system access)
- ✅ **Flask REST API** - 5 endpoints for querying, execution, preview
- ✅ **Streamlit web interface** - beautiful UI with visualizations
- ✅ **Interactive charts** - bar charts, pie charts, line charts
- ✅ **Query history** - track and replay previous queries
- ✅ **Entity browser** - explore all 48 attributes

---

## 🔧 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Disk Space**: 10GB for models and data
- **GPU**: Optional but recommended (2-3x faster)

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install poppler-utils tesseract-ocr

# macOS
brew install poppler tesseract

# Windows
# Download and install:
# - Poppler: http://blog.alivate.com.au/poppler-windows/
# - Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
```

---

## 💿 Installation

### 1. Clone or Download Project Files

Ensure you have these files:
- `contract_generator.py`
- `extract_text.py`
- `extract_attributes.py`
- `query_engine.py`
- `contract_query_app.py`
- `requirements.txt`

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **ML/AI**: PyTorch, Transformers, Sentence-Transformers
- **PDF**: ReportLab, PyPDF2, pdf2image, Pillow, pytesseract
- **Vector Search**: FAISS
- **Data**: pandas, numpy
- **API**: Flask, Flask-CORS
- **Frontend**: Streamlit, Plotly

### 3. Verify Installation

```bash
python -c "import torch, transformers, faiss, streamlit; print('All imports successful!')"
```

---

## 🚀 Quick Start

### Complete Pipeline (5 Steps)

```bash
# Step 1: Generate 100 contract PDFs (~1-1.5 hours)
python contract_generator.py

# Step 2: Extract text from PDFs (~2-50 hours depending on method)
python extract_text.py

# Step 3: Extract entities to CSV (~3-5 hours)
python extract_attributes.py

# Step 4: Start Flask API (Terminal 1)
python query_engine.py

# Step 5: Start Streamlit App (Terminal 2)
streamlit run contract_query_app.py
```

### Query System Only (if you have the CSV)

```bash
# Terminal 1: Start API
python query_engine.py

# Terminal 2: Start Streamlit
streamlit run contract_query_app.py

# Open browser at: http://localhost:8501
```

---

## 📖 Detailed Usage

### Step 1: Generate Contract PDFs

```bash
python contract_generator.py
```

**Output:**
- 100 PDF files in `/mnt/user-data/outputs/`
- Each PDF: 30+ pages, 300+ words per page
- Contract types: MSA, SOW, NDA, Employment, SaaS, etc.

**Example output:**
```
contract_001_MSA_TechNova_Solutions.pdf
contract_002_SOW_Pinnacle_Manufacturing.pdf
contract_003_NDA_Quantum_Analytics.pdf
...
```

**Time:** ~1-1.5 hours for 100 contracts

---

### Step 2: Extract Text from PDFs

```bash
python extract_text.py
```

**How it works:**
1. Converts each PDF page to an image
2. Tries BLIP-2 vision model (most accurate)
3. Falls back to OCR if vision model fails
4. Falls back to direct extraction if OCR fails

**Output:**
- Text files in `/mnt/user-data/outputs/extracted_texts/`
- One .txt file per PDF

**Time:**
- With vision model: ~15-30 min per PDF (slow but accurate)
- With OCR fallback: ~2-3 min per PDF (faster)
- With direct extraction: ~10 sec per PDF (fastest)

**Configuration:**
```python
# In extract_text.py, change:
USE_GPU = True  # Use GPU if available (2-3x faster)
```

---

### Step 3: Extract Contract Entities

```bash
python extract_attributes.py
```

**How it works:**
1. Chunks each text file (500 words with 100-word overlap)
2. Creates FAISS vector index for semantic search
3. For each of 48 entities:
   - Retrieves top-3 most relevant chunks
   - Uses Flan-T5 to extract entity value
4. Outputs all results to CSV

**Output:**
- `contract_entities.csv` with 100 rows × 49 columns
- Columns: filename + 48 entity attributes

**Time:** ~2-3 minutes per contract (~3-5 hours for 100 contracts)

**Extracted entities:**
- party_a_name, party_b_name
- contract_type, effective_date, expiration_date
- total_contract_value, payment_terms
- governing_law, dispute_resolution
- ip_ownership, confidentiality_term
- ... and 38 more!

---

### Step 4: Query System Setup

#### 4.1 Start Flask API (Terminal 1)

```bash
python query_engine.py
```

**Output:**
```
CONTRACT QUERY API
LLM-Powered Pandas Code Generation
================================================================================
Loading data...
✓ Loaded contracts CSV: (100, 49)

Initializing LLM...
Initializing Code Generation LLM on cpu...
  Loading Salesforce/codegen-350M-mono...
  ✓ Model loaded successfully!

API SERVER READY
Available endpoints:
  GET  /health     - Health check
  GET  /entities   - Get available entities
  POST /query      - Process natural language query
  POST /execute    - Execute custom pandas code
  GET  /preview    - Preview contract data
```

API runs on: `http://localhost:5000`

#### 4.2 Start Streamlit Frontend (Terminal 2)

```bash
streamlit run contract_query_app.py
```

Browser opens automatically at: `http://localhost:8501`

---

### Step 5: Query Your Contracts

#### Example Queries (in Streamlit):

**Financial Analysis:**
```
What is the total contract value grouped by contract type?
Show me all contracts with payment terms of Net 30 or less
Which party has the highest total contract value as Party A?
```

**Risk & Compliance:**
```
List all contracts that expire in the next 6 months
Show me contracts without a liability cap specified
Which contracts have non-compete clauses?
```

**Operational Insights:**
```
How many contracts use arbitration versus litigation?
Show me all SaaS agreements with their payment amounts
Which contracts are governed by California law?
```

**Strategic Planning:**
```
Find all contracts with TechNova Solutions as Party B and show their total value
Show me all MSA contracts signed in 2024
List contracts with highest payment amounts
```

#### Using Custom Pandas Code:

In the **Custom Code** tab:

```python
# Filter MSA contracts
result = df[df['contract_type'] == 'MSA']

# Count by contract type
result = df['contract_type'].value_counts()

# Group by party and sum values
result = df.groupby('party_a_name')['total_contract_value'].sum().sort_values(ascending=False)

# Complex filtering
result = df[(df['contract_type'] == 'MSA') & 
            (df['governing_law'].str.contains('California', na=False))]
```

---

## 📁 Project Structure

```
contract-processing-system/
│
├── contract_generator.py          # Step 1: Generate PDFs
├── extract_text.py                # Step 2: Extract text from PDFs
├── extract_attributes.py          # Step 3: Extract entities to CSV
├── query_engine.py                # Step 4: Flask API backend
├── contract_query_app.py          # Step 4: Streamlit frontend
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── COMMANDS.sh                    # Complete command reference
│
└── outputs/                       # Generated files
    ├── contract_*.pdf             # 100 contract PDFs
    ├── extracted_texts/
    │   └── contract_*.txt         # Extracted text files
    └── contract_entities.csv      # Final CSV with all entities
```

---

## 🔌 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. GET `/health`
Check API health status

**Response:**
```json
{
  "status": "healthy",
  "csv_loaded": true,
  "llm_loaded": true,
  "rows": 100,
  "columns": 49
}
```

#### 2. GET `/entities`
Get list of all 48 contract entities

**Response:**
```json
{
  "entities": {
    "party_a_name": "Name of the first party (Party A, Provider, Vendor)",
    "party_b_name": "Name of the second party (Party B, Client, Customer)",
    ...
  },
  "count": 48
}
```

#### 3. POST `/query`
Process natural language query

**Request:**
```json
{
  "query": "Show me all MSA contracts"
}
```

**Response:**
```json
{
  "success": true,
  "query": "Show me all MSA contracts",
  "generated_code": "result = df[df['contract_type'] == 'MSA']",
  "result": {
    "type": "dataframe",
    "data": [...],
    "columns": [...],
    "shape": [10, 49]
  },
  "execution_time": 1.23
}
```

#### 4. POST `/execute`
Execute custom pandas code

**Request:**
```json
{
  "code": "result = df['contract_type'].value_counts()"
}
```

**Response:**
```json
{
  "success": true,
  "code": "result = df['contract_type'].value_counts()",
  "result": {
    "type": "series",
    "data": {"MSA": 20, "SOW": 18, ...}
  }
}
```

#### 5. GET `/preview?limit=10`
Preview contract data

**Response:**
```json
{
  "success": true,
  "data": [...],
  "columns": [...],
  "total_rows": 100,
  "preview_rows": 10
}
```

---

## ⚙️ Configuration

### PDF Generation Settings

Edit `contract_generator.py`:

```python
# Number of contracts to generate
for i in range(1, 101):  # Change 101 to desired number + 1

# Use GPU for faster generation
generator = ContractGenerator(use_gpu=True)
```

### Text Extraction Settings

Edit `extract_text.py`:

```python
# Input/Output folders
INPUT_FOLDER = "/path/to/pdfs"
OUTPUT_FOLDER = "/path/to/output"

# Use GPU
USE_GPU = True

# Skip vision model (use OCR/direct extraction only)
self.vision_model_available = False  # Set in load_vision_model()
```

### Entity Extraction Settings

Edit `extract_attributes.py`:

```python
# Input/Output paths
TEXT_FILES_FOLDER = "/path/to/text/files"
OUTPUT_CSV = "/path/to/output.csv"

# Use GPU
USE_GPU = True

# Adjust chunking
chunk_size = 500    # Words per chunk
overlap = 100       # Word overlap

# Adjust retrieval
top_k = 3          # Number of chunks to retrieve per entity
```

### Query System Settings

Edit `query_engine.py`:

```python
# CSV path
CSV_PATH = "/path/to/contract_entities.csv"

# Use GPU
use_gpu = True  # In initialize_llm()

# API port
app.run(host='0.0.0.0', port=5000)  # Change port here
```

Edit `contract_query_app.py`:

```python
# API URL
API_BASE_URL = "http://localhost:5000"  # Change if using different port
```

---

## 🐛 Troubleshooting

### Issue: Port Already in Use

```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Kill process on port 8501
lsof -ti:8501 | xargs kill -9

# Or run Streamlit on different port
streamlit run contract_query_app.py --server.port 8502
```

### Issue: API Not Connecting

1. Check Flask API is running (Terminal 1)
2. Check for errors in Terminal 1
3. Test API health:
   ```bash
   curl http://localhost:5000/health
   ```

### Issue: CUDA Out of Memory

```python
# Set USE_GPU = False in all scripts
USE_GPU = False
```

### Issue: Vision Model Too Slow

In `extract_text.py`, skip vision model:

```python
def load_vision_model(self):
    # Comment out model loading
    self.vision_model_available = False
    print("Skipping vision model, using OCR/direct extraction")
```

### Issue: Missing System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils tesseract-ocr

# Check if installed
which pdftoppm
which tesseract
```

### Issue: CSV Not Found

```bash
# Check if CSV exists
ls -lh /mnt/user-data/outputs/contract_entities.csv

# If not, run entity extraction
python extract_attributes.py
```

---

## ⚡ Performance Optimization

### Use GPU Acceleration

```python
# In all scripts, set:
USE_GPU = True  # or use_gpu=True
```

**Speed improvement:** 2-3x faster

### Parallel Processing

For Step 2 & 3, split PDFs across multiple machines:

```bash
# Machine 1: Process PDFs 1-33
# Machine 2: Process PDFs 34-66
# Machine 3: Process PDFs 67-100
```

### Skip Vision Model

For faster text extraction (with slight accuracy loss):

```python
# In extract_text.py
self.vision_model_available = False
```

**Time reduction:** 15-30 min → 2-3 min per PDF

### Reduce Number of Contracts

For testing, generate fewer contracts:

```python
# In contract_generator.py
for i in range(1, 11):  # Generate only 10 contracts
```

### Batch Processing

For entity extraction, process in batches:

```python
# Process first 10 files
text_files = list(self.text_files_folder.glob("*.txt"))[:10]
```

---

## 🔒 Security

### Safe Code Execution

The query system uses a **restricted execution environment**:

#### ✅ Allowed:
- pandas, numpy operations
- Basic Python builtins (len, str, int, max, min, etc.)
- datetime operations

#### ❌ Blocked:
- File I/O (open, read, write)
- Network access (requests, urllib, socket)
- System commands (os, sys, subprocess)
- Dynamic code execution (eval, exec, compile)
- Imports of any modules

### Code Validation

All generated code is validated before execution:

```python
FORBIDDEN_KEYWORDS = [
    'import', 'exec', 'eval', 'compile', '__import__',
    'open', 'file', 'input', 'raw_input',
    'os.', 'sys.', 'subprocess', 'shutil',
    'pickle', 'socket', 'urllib', 'requests',
]
```

### Data Isolation

- API works on a **copy** of the DataFrame
- Original data is never modified
- No persistence of user queries or results

---

## 🎯 Use Cases

### 1. Legal Tech & Contract Management
- Automated contract analysis
- Due diligence automation
- Contract portfolio analytics
- Risk assessment

### 2. M&A and Corporate Development
- Contract discovery and cataloging
- Obligation extraction
- Liability assessment
- Compliance checking

### 3. Procurement & Vendor Management
- Vendor contract tracking
- Payment term analysis
- Renewal management
- Spend analytics

### 4. Legal Research & Training
- Contract clause analysis
- Legal language patterns
- Training data for legal AI
- Contract template generation

---

## 📊 Expected Performance

### Time Estimates

| Step | Task | Time (100 contracts) |
|------|------|---------------------|
| 1 | PDF Generation | 1-1.5 hours |
| 2 | Text Extraction (Vision) | 25-50 hours |
| 2 | Text Extraction (OCR) | 2-3 hours |
| 2 | Text Extraction (Direct) | 15-20 minutes |
| 3 | Entity Extraction | 3-5 hours |
| 4 | Query System Startup | 30 seconds |
| **Total (Vision)** | **Complete Pipeline** | **30-60 hours** |
| **Total (OCR)** | **Complete Pipeline** | **6-10 hours** |

### Resource Usage

| Component | CPU | RAM | Disk |
|-----------|-----|-----|------|
| PDF Generation | ~50% | 2-4 GB | 500 MB |
| Text Extraction | ~70% | 4-8 GB | 1 GB |
| Entity Extraction | ~80% | 6-10 GB | 100 MB |
| Query API | ~30% | 2-4 GB | Minimal |
| Streamlit | ~20% | 500 MB | Minimal |

---


1. **Additional Contract Types**: Add more contract templates
2. **More Entities**: Expand the 48 attributes to include more fields
3. **Better LLMs**: Experiment with larger/better code generation models
4. **UI Enhancements**: Improve Streamlit interface
5. **Performance**: Optimize chunking and retrieval strategies
6. **Multilingual Support**: Add support for non-English contracts
7. **Advanced Analytics**: Add more visualization types


## Acknowledgments

This project uses the following open-source technologies:

- **Hugging Face Transformers** - LLM infrastructure
- **Google Flan-T5** - Text generation and extraction
- **Salesforce BLIP-2** - Vision-language model
- **Salesforce CodeGen** - Code generation
- **FAISS** - Vector similarity search
- **Flask** - REST API framework
- **Streamlit** - Web interface
- **PyTorch** - Deep learning framework
- **ReportLab** - PDF generation
- **Tesseract** - OCR engine

---


**Quick test with minimal contracts:**

```bash
# 1. Install dependencies
pip install -r requirements.txt
sudo apt-get install poppler-utils tesseract-ocr  # Linux

# 2. Generate 5 test contracts (edit contract_generator.py: range(1, 6))
python contract_generator.py

# 3. Extract text (will use fallback methods if vision model is slow)
python extract_text.py

# 4. Extract entities
python extract_attributes.py

# 5. Start query system
# Terminal 1:
python query_engine.py

# Terminal 2:
streamlit run contract_query_app.py

# 6. Query in browser: http://localhost:8501
```
