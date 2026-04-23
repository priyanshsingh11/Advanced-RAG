from qdrant_client import models
from fastembed import TextEmbedding, SparseTextEmbedding
from app.db.qdrant_store import QdrantStore
from app.core.config import settings
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, store: QdrantStore):
        self.store = store
        self.client = store.get_client()
        
        # Initialize embedding models
        # These will be downloaded on first use
        self.dense_model = TextEmbedding(model_name=settings.DENSE_EMBEDDING_MODEL)
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25") # Optimized BM25 for Qdrant

    def retrieve(self, query: str, top_k: int = 20, hyde_query: str = None, filters: List[str] = None) -> List[Dict[str, Any]]:
        """Performs hybrid search (Dense + Sparse) and returns top K documents."""
        try:
            # 1. Generate Embeddings
            dense_search_text = hyde_query if hyde_query else query
            dense_vector = list(self.dense_model.embed([dense_search_text]))[0].tolist()
            sparse_result = list(self.sparse_model.embed([query]))[0]
            
            # 2. Build Qdrant Filter if metadata filters are provided
            qdrant_filter = None
            if filters:
                conditions = []
                for f in filters:
                    # Use MatchText for flexible matching (e.g., "AI" matching "data\AI.pdf")
                    conditions.append(models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchText(text=f)
                    ))
                qdrant_filter = models.Filter(should=conditions)
                logger.info(f"Applying metadata filters: {filters}")

            # 3. Perform Hybrid Search using Qdrant's Query API (RRF Fusion)
            results = self.client.query_points(
                collection_name=settings.COLLECTION_NAME,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=top_k,
                        filter=qdrant_filter # Apply filter to dense prefetch
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_result.indices.tolist(),
                            values=sparse_result.values.tolist()
                        ),
                        using="sparse",
                        limit=top_k,
                        filter=qdrant_filter # Apply filter to sparse prefetch
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True
            ).points

            # 3. Format results
            documents = []
            for point in results:
                # Small-to-Big expansion: If a parent exists, use it as context for the LLM
                metadata = point.payload.get("metadata", {})
                parent_text = metadata.get("parent_text")
                content = parent_text if parent_text else point.payload.get("text", "")
                
                documents.append({
                    "id": point.id,
                    "content": content,
                    "metadata": metadata,
                    "score": point.score
                })
            
            return documents

        except Exception as e:
            logger.error(f"Error in HybridRetriever: {e}")
            return []
