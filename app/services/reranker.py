from sentence_transformers import CrossEncoder
from app.core.config import settings
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self):
        # Initialize the BGE Cross-Encoder
        # This will be downloaded on first use
        try:
            self.model = CrossEncoder(settings.RERANKER_MODEL)
        except Exception as e:
            logger.error(f"Failed to load Reranker model: {e}")
            self.model = None

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Re-scores documents based on the query using a Cross-Encoder."""
        if not self.model or not documents:
            return documents[:top_k]

        try:
            # Prepare pairs: [query, doc_content]
            pairs = [[query, doc["content"]] for doc in documents]
            
            # Get scores
            scores = self.model.predict(pairs)
            
            # Attach scores to documents
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)
            
            # Sort by rerank score descending
            reranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
            
            return reranked_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return documents[:top_k]
