from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from typing import List
import os
import logging

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNKING_SIZE,
            chunk_overlap=settings.CHUNKING_OVERLAP,
            length_function=len,
            add_start_index=True,
        )

    def load_and_split(self, data_path: str = "./data") -> List:
        """Loads PDFs and Text files from a directory and splits them into chunks."""
        documents = []
        
        if not os.path.exists(data_path):
            logger.error(f"Data path {data_path} does not exist.")
            return []

        try:
            # Load PDFs
            pdf_loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            
            # Load Text files
            txt_loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader)
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)
            
            if not documents:
                logger.warning("No documents found in the data directory.")
                return []

            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Loaded {len(documents)} documents and split into {len(chunks)} chunks.")
            
            return chunks

        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []
