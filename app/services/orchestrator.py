from app.services.query_rewriter import QueryRewriter
from app.services.hybrid_retriever import HybridRetriever
from app.services.reranker import Reranker
from app.services.generator import Generator
from app.db.qdrant_store import QdrantStore
from app.core.config import settings
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class RAGOrchestrator:
    def __init__(self):
        # Initialize all services
        self.store = QdrantStore()
        self.rewriter = QueryRewriter()
        self.retriever = HybridRetriever(self.store)
        self.reranker = Reranker()
        self.generator = Generator()

    def query(self, user_query: str) -> Dict[str, Any]:
        """Orchestrates the full Advanced RAG pipeline using local Ollama."""
        try:
            logger.info(f"Starting RAG pipeline for query: {user_query}")
            
            # 1. Query Rewriting (Optimizing for Hybrid Search)
            rewritten_query = self.rewriter.rewrite(user_query)
            logger.info(f"Rewritten Query: {rewritten_query}")
            
            # 2. Hybrid Retrieval (Dense + Sparse) - Top 20
            retrieved_docs = self.retriever.retrieve(rewritten_query, top_k=settings.TOP_K_RETRIEVAL)
            logger.info(f"Retrieved {len(retrieved_docs)} documents.")
            
            if not retrieved_docs:
                return {
                    "answer": "Context not found in documents.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # 3. Reranking (Cross-Encoder) - Top 5
            reranked_docs = self.reranker.rerank(rewritten_query, retrieved_docs, top_k=settings.TOP_K_RERANK)
            logger.info(f"Reranking complete. Top score: {reranked_docs[0].get('rerank_score', 0):.4f}")
            
            # 4. LLM Generation (Local Ollama Llama 3)
            result = self.generator.generate(user_query, reranked_docs)
            
            return result

        except Exception as e:
            logger.error(f"Error in RAGOrchestrator: {e}")
            return {
                "answer": f"A pipeline error occurred: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
