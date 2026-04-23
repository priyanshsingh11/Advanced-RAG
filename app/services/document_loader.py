from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings
from typing import List
import os
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    def __init__(self):
        # 1. Parent Splitter (Large Context)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            add_start_index=True,
        )

        # 2. Child Splitter (Small Precise Chunks)
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            add_start_index=True,
        )

        # 3. Semantic Splitter (Optional enhancement)
        if settings.USE_SEMANTIC_CHUNKING:
            self.embeddings = HuggingFaceEmbeddings(model_name=settings.DENSE_EMBEDDING_MODEL)
            self.semantic_splitter = SemanticChunker(self.embeddings)
        else:
            self.semantic_splitter = None

    def load_and_split(self, data_path: str = "./data", file_path: str = None) -> List:
        """Loads documents and splits them using a Parent-Child strategy."""
        documents = []
        
        try:
            if file_path:
                if not os.path.exists(file_path):
                    logger.error(f"File not found: {file_path}")
                    return []
                
                if file_path.lower().endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif file_path.lower().endswith(".txt"):
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
                else:
                    logger.error(f"Unsupported file type: {file_path}")
                    return []
            else:
                if not os.path.exists(data_path):
                    return []
                # Load PDFs and TXTs
                pdf_loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
                documents.extend(pdf_loader.load())
                txt_loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader)
                documents.extend(txt_loader.load())

            if not documents:
                return []

            # Clean documents
            cleaned_docs = []
            for doc in documents:
                text = doc.page_content.strip()
                if len(text) > 50:
                    doc.page_content = text.encode("utf-8", "ignore").decode("utf-8")
                    cleaned_docs.append(doc)

            # --- Parent-Child Chunking Strategy ---
            final_chunks = []
            
            # 1. Create Parent Chunks
            parent_docs = self.parent_splitter.split_documents(cleaned_docs)
            logger.info(f"Created {len(parent_docs)} Parent documents.")

            # 2. For each Parent, create smaller Children
            for i, parent in enumerate(parent_docs):
                parent_id = f"p_{i}"
                parent_text = parent.page_content
                
                # Split the parent into smaller children
                child_docs = self.child_splitter.split_documents([parent])
                
                for child in child_docs:
                    # Enrich child metadata with parent info
                    child.metadata["parent_id"] = parent_id
                    child.metadata["parent_text"] = parent_text # This allows instant expansion during retrieval
                    final_chunks.append(child)

            logger.info(f"Generated {len(final_chunks)} Child chunks with Parent context.")
            return final_chunks

        except Exception as e:
            logger.error(f"Error in Parent-Child chunking: {e}")
            return []