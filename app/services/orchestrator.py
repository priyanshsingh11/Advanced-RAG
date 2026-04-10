from app.services.hybrid_retriever import HybridRetriever
from app.services.reranker import Reranker
from app.db.qdrant_store import QdrantStore
from app.core.config import settings
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class RetrievalPipeline:
    def __init__(self):
        # Initialize retrieval services
        self.store = QdrantStore()
        self.retriever = HybridRetriever(self.store)
        self.reranker = Reranker()

    def run(self, user_query: str) -> List[Dict[str, Any]]:
        """Orchestrates the retrieval and reranking pipeline."""
        try:
            logger.info(f"Starting retrieval pipeline for query: {user_query}")
            
            # 1. Hybrid Retrieval (Vector + BM25) - Top 20
            retrieved_docs = self.retriever.retrieve(user_query, top_k=settings.TOP_K_RETRIEVAL)
            logger.info(f"Retrieved {len(retrieved_docs)} documents.")
            
            if not retrieved_docs:
                return []
            
            # 2. Reranking (Cross-Encoder) - Top 5
            reranked_docs = self.reranker.rerank(user_query, retrieved_docs, top_k=settings.TOP_K_RERANK)
            logger.info(f"Reranking complete. Top score: {reranked_docs[0].get('rerank_score', 0):.4f}")
            
            return reranked_docs

        except Exception as e:
            logger.error(f"Error in RetrievalPipeline: {e}")
            return []
