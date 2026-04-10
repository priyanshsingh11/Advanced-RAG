# Advanced Retrieval-Augmented Generation (RAG) Backend

A high-performance, production-grade retrieval pipeline designed for state-of-the-art Document AI applications. This system implements a modular multi-stage retrieval architecture using hybrid search (Dense + Sparse) and Cross-Encoder reranking.

## System Architecture

The project follows a modular RAG pipeline designed for maximum precision and recall:

1. **Document Ingestion Layer**: utilizes LangChain's directory loaders and recursive character splitting to process heterogeneous data sources (PDF, TXT).
2. **Hybrid Embedding Generation**:
    - **Dense Vectors**: Employs `BAAI/bge-base-en-v1.5` for deep semantic understanding.
    - **Sparse Vectors**: Generates BM25-compatible sparse embeddings for precise keyword matching.
3. **Vector Infrastructure (Qdrant)**: A high-performance vector database that manages dual-indexing (Dense + Sparse) and executes Reciprocal Rank Fusion (RRF) at the database level for optimized hybrid results.
4. **Ranking Refinement (Cross-Encoder)**: Implements a second-stage reranker using `BAAI/bge-reranker-base` to mitigate "lost in the middle" phenomena and ensure only the most relevant context reaches the generation stage.

## Technical Components

- **FastAPI**: Asynchronous Python framework for high-concurrency API performance.
- **Qdrant**: Vector search engine with native support for hybrid search and persistent disk storage.
- **FastEmbed**: Optimized inference library for BGE and BM25 embeddings, reducing latency and resource overhead.
- **Sentence-Transformers**: Powering the Cross-Encoder reranking stage.
- **Pydantic V2**: Robust data validation and settings management.

## Project Structure

```text
Advanced RAG/
├── app/
│   ├── api/          # Asynchronous endpoint definitions
│   ├── core/         # System configuration and global settings
│   ├── db/           # Database connection and collection management
│   ├── services/     # Modular pipeline components (Retrieval, Reranking, Loading)
│   └── main.py       # Application entry point
├── data/             # Persistent storage for raw source documents
├── storage/          # Local Qdrant database storage
├── .env              # Environment-specific configuration
├── requirements.txt  # Dependency specifications
└── README.md         # System documentation
```

## Setup and Installation

### 1. Environment Initialization
Initialize a isolated Python environment to manage dependencies:

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 2. Dependency Management
Install the required production and inference libraries:

```bash
pip install -r requirements.txt
```

### 3. Configuration
Configure the `.env` file with appropriate API keys and model identifiers. The system is designed to be model-agnostic at the generation layer.

## Operational Workflow

### Retrieval Pipeline
The central orchestration is handled by the `RetrievalPipeline` class, which executes a non-linear search strategy:
1. **Parallel Search**: Executes a concurrent dense semantic search and sparse keyword search in Qdrant.
2. **Database-Level Fusion**: Utilizes RRF to normalize and combine disparate score distributions.
3. **Contextual Reranking**: Passes the top fusion results (default k=20) through a Cross-Encoder to verify relevance against the original query.
4. **Selection**: Returns the top refined results (default k=5) for downstream generation.

## Performance Optimization

- **Local Persistence**: Qdrant is configured in persistence mode, allowing for rapid restarts without re-indexing.
- **Inference Caching**: Embedding and reranker models are cached locally using `fastembed` and `sentence-transformers` protocols.
- **Chunking Strategy**: Employs recursive splitting with configurable overlap to maintain semantic continuity across document fragments.

## License
This project is licensed under the MIT License.
