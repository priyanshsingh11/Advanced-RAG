import contextlib
from fastapi import FastAPI
from app.api.endpoints import router
from app.core.config import settings
from app.db.qdrant_store import QdrantStore
from app.services.orchestrator import RAGOrchestrator
from app.services.document_loader import DocumentLoader
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize heavy resources
    logger.info("Initializing RAG resources...")
    store = QdrantStore()
    loader = DocumentLoader()
    orchestrator = RAGOrchestrator(store=store)
    
    # Store in app state for access in routes
    app.state.store = store
    app.state.loader = loader
    app.state.orchestrator = orchestrator
    
    logger.info("RAG resources initialized successfully.")
    yield
    # Shutdown: Clean up if needed
    logger.info("Shutting down RAG resources...")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title=settings.APP_NAME,
    description="Advanced RAG Backend with Hybrid Search and Local LLM (Ollama)",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "Advanced RAG API is running",
        "docs": "/docs",
        "status": "online"
    }

if __name__ == "__main__":
    import uvicorn
    # On Windows, reload=True can cause double-initialization of global resources.
    # When using lifespan and running as a module, we set reload to False for stability.
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
