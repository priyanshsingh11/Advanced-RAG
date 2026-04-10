from qdrant_client import QdrantClient, models
from app.core.config import settings
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
                            index=models.SparseIndexParams(on_disk=True) # BM25 index
                        )
                    }
                )
            else:
                logger.info(f"Collection {self.collection_name} already exists.")
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {e}")

    def get_client(self) -> QdrantClient:
        return self.client
