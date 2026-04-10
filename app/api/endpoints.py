from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.schemas.query import QueryRequest, QueryResponse, IngestResponse
from app.services.orchestrator import RAGOrchestrator
from app.services.document_loader import DocumentLoader
from app.db.qdrant_store import QdrantStore
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize orchestrator and store at module level for persistence
orchestrator = RAGOrchestrator()
loader = DocumentLoader()
store = QdrantStore()

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Executes a RAG query using the modular pipeline."""
    try:
        result = orchestrator.query(request.query)
        if not result["sources"]:
            logger.warning(f"No documents found for query: {request.query}")
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents():
    """Triggers document ingestion from the /data folder."""
    try:
        # 1. Load and split documents
        chunks = loader.load_and_split()
        if not chunks:
            return IngestResponse(
                message="No documents found or processed in the /data folder.",
                status="warning",
                chunks_processed=0
            )

        # 2. Upsert to Qdrant
        success = store.upsert_documents(chunks)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to upload documents to vector store.")

        return IngestResponse(
            message=f"Successfully processed and indexed {len(chunks)} chunks.",
            status="success",
            chunks_processed=len(chunks)
        )
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
