import os
from fastembed import TextEmbedding, SparseTextEmbedding
from sentence_transformers import CrossEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    # 1. Dense Embedding Model (fastembed)
    dense_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    logger.info(f"Downloading Dense Embedding model: {dense_model_name}")
    TextEmbedding(model_name=dense_model_name)
    
    # 2. Sparse Embedding Model (fastembed)
    sparse_model_name = "Qdrant/bm25"
    logger.info(f"Downloading Sparse Embedding model: {sparse_model_name}")
    SparseTextEmbedding(model_name=sparse_model_name)
    
    # 3. Reranker Model (sentence-transformers)
    reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    logger.info(f"Downloading Reranker model: {reranker_model_name}")
    CrossEncoder(reranker_model_name)
    
    logger.info("All models downloaded successfully.")

if __name__ == "__main__":
    download_models()
