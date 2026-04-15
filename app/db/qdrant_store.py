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

    def upsert_documents(self, chunks: List[Any], batch_size: int = 500):
        """Embeds and uploads chunks to Qdrant in batches to optimize memory usage."""
        try:
            total_chunks = len(chunks)
            total_batches = (total_chunks + batch_size - 1) // batch_size
            
            logger.info(f"Starting ingestion of {total_chunks} chunks in {total_batches} batches.")

            for i in range(0, total_chunks, batch_size):
                batch_idx = i // batch_size + 1
                batch_chunks = chunks[i : i + batch_size]
                
                # SANITIZATION: Aggressive cleaning of text content
                valid_batch_chunks = []
                for chunk in batch_chunks:
                    try:
                        content = getattr(chunk, 'page_content', "")
                        if content is None:
                            continue
                        
                        # Force string, strip whitespace, and normalize UTF-8 (removes null bytes/broken chars)
                        clean_content = str(content).encode("utf-8", "ignore").decode("utf-8").strip()
                        
                        if clean_content:
                            chunk.page_content = clean_content
                            valid_batch_chunks.append(chunk)
                    except Exception as e:
                        logger.warning(f"Failed to sanitize a chunk in batch {batch_idx}: {e}")
                
                if not valid_batch_chunks:
                    logger.warning(f"Batch {batch_idx} contained no valid chunks after sanitization. Skipping.")
                    continue

                texts = [chunk.page_content for chunk in valid_batch_chunks]
                metadatas = [chunk.metadata for chunk in valid_batch_chunks]
                
                logger.info(f"Processing batch {batch_idx}/{total_batches} ({len(valid_batch_chunks)} sanitized chunks)...")
                
                try:
                    # Generate Embeddings for current batch
                    dense_vectors = list(self.dense_model.embed(texts))
                    sparse_vectors = list(self.sparse_model.embed(texts))
                except Exception as e:
                    logger.error(f"EMBEDDING ERROR in Batch {batch_idx}: {e}")
                    # FORENSICS: Print first few characters of each text in the failing batch
                    for idx, t in enumerate(texts):
                        preview = (t[:100] + '...') if len(t) > 100 else t
                        logger.error(f"  Chunk {idx} (len={len(t)}): {preview!r}")
                    raise e # Re-raise to stop and let user see the forensic logs
                
                points = []
                for j, (text, meta, dense, sparse) in enumerate(zip(texts, metadatas, dense_vectors, sparse_vectors)):
                    # Unique ID based on content hash and global index
                    point_id = hash(text + str(i + j)) % (10**15)
                    points.append(
                        models.PointStruct(
                            id=point_id,
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
                logger.info(f"Successfully upserted batch {batch_idx}/{total_batches}.")

            return True
        except Exception as e:
            logger.error(f"Error upserting to Qdrant: {e}")
            return False


    def get_client(self) -> QdrantClient:
        return self.client
