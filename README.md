# Advanced RAG Backend

A production-quality modular Retrieval-Augmented Generation (RAG) system built with FastAPI, Qdrant, and Hugging Face models.

## 🚀 Roadmap
1. **[User Query]**
2. **[Query Rewriter]**: Rephrases query for better retrieval.
3. **[Hybrid Retriever]**: Dense (BGE-base) + Sparse (BM25) search via Qdrant.
4. **[Cross-Encoder Ranker]**: Reranks top 20 docs into top 5 using BGE-reranker.
5. **[Context Filter]**: Removes irrelevant blocks.
6. **[LLM Generator]**: Final answer generation with sources and confidence scores.

## 🛠️ Tech Stack
- **Framework**: FastAPI
- **Vector DB**: Qdrant
- **Embeddings**: BAAI/bge-base-en-v1.5
- **Reranker**: BAAI/bge-reranker-base
- **Orchestration**: LangChain / Custom Modular Logic

## 📂 Project Structure
```text
Advanced RAG/
├── app/
│   ├── api/          # FastAPI Endpoints
│   ├── core/         # Config and constants
│   ├── db/           # Qdrant connection and logic
│   ├── services/     # Modular RAG services (Loader, Retriever, Reranker, etc.)
│   └── main.py       # Application entry point
├── data/             # Document storage for ingestion
├── storage/          # Local Qdrant data (Git ignored)
├── .env              # Environment variables
├── requirements.txt  # Project dependencies
└── README.md
```

## ⚙️ Setup Instructions

### 1. Clone & Initialize
```bash
git clone https://github.com/priyanshsingh11/Advanced-RAG.git
cd "Advanced RAG"
```

### 2. Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration
Create a `.env` file based on the provided template and add your API keys.

### 5. Run the Project
```bash
uvicorn app.main:app --reload
```

## 📝 License
MIT
