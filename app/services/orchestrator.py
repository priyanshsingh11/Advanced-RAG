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
    def __init__(self, store: QdrantStore = None):
        # Initialize all services
        self.store = store or QdrantStore()
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

    def compare(self, user_query: str) -> Dict[str, Any]:
        """Runs the RAG pipeline and compares multiple LLM backends."""
        try:
            logger.info(f"Starting model comparison for query: {user_query}")
            
            # 1. Processing (Same for all models)
            rewritten_query = self.rewriter.rewrite(user_query)
            retrieved_docs = self.retriever.retrieve(rewritten_query, top_k=settings.TOP_K_RETRIEVAL)
            
            if not retrieved_docs:
                return {"query": user_query, "results": [], "metadata": {"error": "No context found"}}
            
            reranked_docs = self.reranker.rerank(rewritten_query, retrieved_docs, top_k=settings.TOP_K_RERANK)

            # 2. Benchmarking (Parallel-ish or Sequential)
            results = []
            
            # Ollama Models
            ollama_models = [m.strip() for m in settings.OLLAMA_MODELS.split(",")]
            for model in ollama_models:
                res = self.generator.generate_with_benchmark(user_query, reranked_docs, model, "ollama")
                results.append(res)
            
            # Groq Model
            if settings.GROQ_API_KEY:
                res = self.generator.generate_with_benchmark(user_query, reranked_docs, settings.GROQ_MODEL, "groq")
                results.append(res)
            else:
                logger.warning("GROQ_API_KEY not found, skipping Groq comparison.")

            return {
                "query": user_query,
                "results": results,
                "metadata": {
                    "rewritten_query": rewritten_query,
                    "docs_retrieved": len(retrieved_docs)
                }
            }

        except Exception as e:
            logger.error(f"Error in RAGOrchestrator Comparison: {e}")
            raise e
