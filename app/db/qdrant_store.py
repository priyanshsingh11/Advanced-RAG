from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from app.core.config import settings
from typing import List, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

class QdrantStore:
    def __init__(self):
        # Initialize client. Local persistent storage mode.
        self.client = QdrantClient(path=settings.QDRANT_PATH)
        self.collection_name = settings.COLLECTION_NAME
        self._init_collection()

    def _init_collection(self):
        """Initializes the collection if it doesn't exist."""
        # Initialize embedding models for ingestion
        self.dense_model = TextEmbedding(model_name=settings.DENSE_EMBEDDING_MODEL)
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=768, # BGE-base-en size
                            distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(
                            index=models.SparseIndexParams(on_disk=True)
                        )
                    }
                )
            else:
                logger.info(f"Collection {self.collection_name} already exists.")
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {e}")

    def upsert_documents(self, chunks: List[Any]):
        """Embeds and uploads chunks to Qdrant."""
        try:
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            # Generate Embeddings
            dense_vectors = list(self.dense_model.embed(texts))
            sparse_vectors = list(self.sparse_model.embed(texts))
            
            points = []
            for i, (text, meta, dense, sparse) in enumerate(zip(texts, metadatas, dense_vectors, sparse_vectors)):
                points.append(
                    models.PointStruct(
                        id=hash(text) % (10**15), # Simple persistent ID
                        vector={
                            "dense": dense.tolist(),
                            "sparse": models.SparseVector(
                                indices=sparse.indices.tolist(),
                                values=sparse.values.tolist()
                            )
                        },
                        payload={
                            "text": text,
                            "metadata": meta
                        }
                    )
                )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Successfully upserted {len(points)} chunks to Qdrant.")
            return True
        except Exception as e:
            logger.error(f"Error upserting to Qdrant: {e}")
            return False

    def get_client(self) -> QdrantClient:
        return self.client
