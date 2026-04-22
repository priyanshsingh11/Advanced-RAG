from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    UnstructuredPDFLoader
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
        # 1. Recursive Splitter (structure-based)
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNKING_SIZE,
            chunk_overlap=settings.CHUNKING_OVERLAP,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""]
        )

        # 2. Semantic Splitter (meaning-based)
        if settings.USE_SEMANTIC_CHUNKING:
            logger.info(f"Initializing SemanticChunker with {settings.DENSE_EMBEDDING_MODEL}")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.DENSE_EMBEDDING_MODEL
            )

            self.semantic_splitter = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type="standard_deviation"  # safer than percentile
            )
        else:
            self.semantic_splitter = None

    def load_and_split(self, data_path: str = "./data") -> List:
        """Loads PDFs and Text files, cleans them, and splits into chunks."""
        
        documents = []

        if not os.path.exists(data_path):
            logger.error(f"Data path {data_path} does not exist.")
            return []

        try:
            # ---------------------------
            # 📄 Load PDFs (better loader)
            # ---------------------------
            pdf_loader = DirectoryLoader(
                data_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)

            # ---------------------------
            # 📄 Load Text Files
            # ---------------------------
            txt_loader = DirectoryLoader(
                data_path,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)

            logger.info(f"Total raw documents loaded: {len(documents)}")

            if not documents:
                logger.warning("No documents found in the data directory.")
                return []

            # ---------------------------
            # 🧹 CLEAN DOCUMENTS (CRITICAL FIX)
            # ---------------------------
            cleaned_documents = []

            for doc in documents:
                try:
                    if doc.page_content and isinstance(doc.page_content, str):
                        text = doc.page_content.strip()

                        # Skip empty or very small text
                        if len(text) > 20:
                            doc.page_content = text
                            cleaned_documents.append(doc)
                        else:
                            logger.warning("Skipping small/empty document chunk")
                    else:
                        logger.warning("Skipping invalid document format")

                except Exception as e:
                    logger.warning(f"Error cleaning document: {e}")

            logger.info(f"Valid documents after cleaning: {len(cleaned_documents)}")

            if not cleaned_documents:
                logger.error("No valid documents after cleaning!")
                return []

            # ---------------------------
            # ✂️ CHUNKING
            # ---------------------------
            if settings.USE_SEMANTIC_CHUNKING and self.semantic_splitter:
                logger.info("Using Semantic Chunking for document splitting...")

                semantic_chunks = self.semantic_splitter.split_documents(cleaned_documents)

                logger.info(f"Semantic chunks created: {len(semantic_chunks)}")

                # Hybrid refinement
                logger.info("Refining semantic chunks with recursive splitting...")
                chunks = self.recursive_splitter.split_documents(semantic_chunks)

            else:
                logger.info("Using Recursive Character splitting...")
                chunks = self.recursive_splitter.split_documents(cleaned_documents)

            logger.info(f"Final chunks generated: {len(chunks)}")

            return chunks

        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []