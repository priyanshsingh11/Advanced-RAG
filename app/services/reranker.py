from sentence_transformers import CrossEncoder
from app.core.config import settings
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self):
        # Initialize cross-encoder/ms-marco-MiniLM-L-6-v2
        # Trained on MS MARCO passage ranking — optimized for query-document relevance scoring
        try:
            self.model = CrossEncoder(settings.RERANKER_MODEL)
            logger.info(f"Reranker loaded: {settings.RERANKER_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load Reranker model: {e}")
            self.model = None

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """Re-scores documents based on the query using a Cross-Encoder."""
        if not self.model or not documents:
            return documents[:top_k]

        try:
            # Prepare pairs: [query, doc_content]
            pairs = [[query, doc["content"]] for doc in documents]
            
            # Get relevance scores from cross-encoder
            scores = self.model.predict(pairs)
            
            # Attach scores while preserving all existing metadata
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)
            
            # Sort by rerank score descending
            reranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
            
            top_docs = reranked_docs[:top_k]
            
            # Log score range for debugging retrieval quality
            if top_docs:
                logger.info(
                    f"Reranked {len(documents)} -> {len(top_docs)} docs | "
                    f"Score range: [{top_docs[-1]['rerank_score']:.4f} — {top_docs[0]['rerank_score']:.4f}]"
                )
            
            return top_docs
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return documents[:top_k]
