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

    def retrieve(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Performs hybrid search (Dense + Sparse) and returns top K documents."""
        try:
            # 1. Generate Embeddings
            dense_vector = list(self.dense_model.embed([query]))[0].tolist()
            
            # Sparse embeddings return as a sparse vector object
            sparse_result = list(self.sparse_model.embed([query]))[0]
            
            # 2. Perform Hybrid Search using Qdrant's Query API (RRF Fusion)
            # Qdrant's .query_points is the modern way to do hybrid search with fusion
            results = self.client.query_points(
                collection_name=settings.COLLECTION_NAME,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=top_k,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_result.indices.tolist(),
                            values=sparse_result.values.tolist()
                        ),
                        using="sparse",
                        limit=top_k,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True
            ).points

            # 3. Format results
            documents = []
            for point in results:
                documents.append({
                    "id": point.id,
                    "content": point.payload.get("text", ""),
                    "metadata": point.payload.get("metadata", {}),
                    "score": point.score
                })
            
            return documents

        except Exception as e:
            logger.error(f"Error in HybridRetriever: {e}")
            return []
