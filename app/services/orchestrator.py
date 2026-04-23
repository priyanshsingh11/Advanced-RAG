from app.services.query_rewriter import QueryRewriter
from app.services.hybrid_retriever import HybridRetriever
from app.services.reranker import Reranker
from app.services.generator import Generator
from app.services.hyde import HyDEGenerator
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
        self.hyde = HyDEGenerator()

    def query(self, user_query: str) -> Dict[str, Any]:
        """Orchestrates the full Advanced RAG pipeline using local Ollama."""
        try:
            logger.info(f"Starting RAG pipeline for query: {user_query}")
            
            # 1. Query Analysis (Rewriting + Filter Extraction)
            analysis = self.rewriter.rewrite(user_query)
            rewritten_query = analysis["rewritten_query"]
            filters = analysis["filters"]
            
            logger.info(f"Rewritten Query: {rewritten_query} | Filters: {filters}")
            
            # 2. HyDE (Hypothetical Document Embedding)
            hyde_query = self.hyde.generate_hypothetical_answer(rewritten_query)
            
            # 3. Hybrid Retrieval with Filters
            retrieved_docs = self.retriever.retrieve(
                query=rewritten_query, 
                hyde_query=hyde_query, 
                filters=filters,
                top_k=settings.TOP_K_RETRIEVAL
            )
            logger.info(f"Retrieved {len(retrieved_docs)} documents.")
            
            if not retrieved_docs:
                return {
                    "answer": "Context not found in documents.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # 4. Reranking (Cross-Encoder)
            reranked_docs = self.reranker.rerank(rewritten_query, retrieved_docs, top_k=settings.TOP_K_RERANK)
            
            # 5. LLM Generation
            result = self.generator.generate(user_query, reranked_docs)
            
            # Metadata for transparency
            result["metadata"] = {
                "rewritten_query": rewritten_query,
                "hyde_query": hyde_query[:50] + "...",
                "filters_applied": filters
            }
            
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
            
            # 1. Processing
            analysis = self.rewriter.rewrite(user_query)
            rewritten_query = analysis["rewritten_query"]
            filters = analysis["filters"]
            
            hyde_query = self.hyde.generate_hypothetical_answer(rewritten_query)
            
            retrieved_docs = self.retriever.retrieve(
                query=rewritten_query, 
                hyde_query=hyde_query, 
                filters=filters,
                top_k=settings.TOP_K_RETRIEVAL
            )
            
            if not retrieved_docs:
                return {"query": user_query, "results": [], "metadata": {"error": "No context found"}}
            
            reranked_docs = self.reranker.rerank(rewritten_query, retrieved_docs, top_k=settings.TOP_K_RERANK)

            # 2. Benchmarking (Sequential)
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

            return {
                "query": user_query,
                "results": results,
                "metadata": {
                    "rewritten_query": rewritten_query,
                    "hyde_query": hyde_query[:100] + "...",
                    "docs_retrieved": len(retrieved_docs)
                }
            }

        except Exception as e:
            logger.error(f"Error in RAGOrchestrator Comparison: {e}")
            raise e
