from fastapi import APIRouter, HTTPException, Request, File, UploadFile
from app.schemas.query import QueryRequest, QueryResponse, IngestResponse, ComparisonResponse
import logging
import os
import shutil

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_rag(request_body: QueryRequest, request: Request):
    """Executes a RAG query using the modular pipeline."""
    try:
        orchestrator = request.app.state.orchestrator
        result = orchestrator.query(request_body.query)
        if not result["sources"]:
            logger.warning(f"No documents found for query: {request_body.query}")
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare", response_model=ComparisonResponse)
async def compare_models(request_body: QueryRequest, request: Request):
    """Executes a RAG query and compares multiple LLMs."""
    try:
        orchestrator = request.app.state.orchestrator
        result = orchestrator.compare(request_body.query)
        return ComparisonResponse(**result)
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: Request):
    """Triggers document ingestion from the /data folder."""
    try:
        loader = request.app.state.loader
        store = request.app.state.store
        
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
@router.post("/upload", response_model=IngestResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """Uploads a document, saves it to /data, and triggers indexing."""
    try:
        loader = request.app.state.loader
        store = request.app.state.store
        
        # 1. Ensure /data directory exists
        data_dir = "./data"
        os.makedirs(data_dir, exist_ok=True)
        
        # 2. Save the file
        file_path = os.path.join(data_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"Saved uploaded file to: {file_path}")
        
        # 3. Load and split JUST this file
        chunks = loader.load_and_split(file_path=file_path)
        if not chunks:
            return IngestResponse(
                message=f"File {file.filename} was uploaded but no content could be extracted.",
                status="warning",
                chunks_processed=0
            )

        # 4. Upsert to Qdrant
        success = store.upsert_documents(chunks)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to upload documents to vector store.")

        return IngestResponse(
            message=f"Successfully uploaded and indexed '{file.filename}' ({len(chunks)} chunks).",
            status="success",
            chunks_processed=len(chunks)
        )
    except Exception as e:
        logger.error(f"Error during file upload and ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
