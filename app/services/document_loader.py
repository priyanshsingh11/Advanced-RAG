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
        # 1. Recursive Splitter
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNKING_SIZE,
            chunk_overlap=settings.CHUNKING_OVERLAP,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""]
        )

        # 2. Semantic Splitter
        if settings.USE_SEMANTIC_CHUNKING:
            logger.info(f"Initializing SemanticChunker with {settings.DENSE_EMBEDDING_MODEL}")

            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.DENSE_EMBEDDING_MODEL
            )

            self.semantic_splitter = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type="standard_deviation"
            )
        else:
            self.semantic_splitter = None

    def load_and_split(self, data_path: str = "./data") -> List:
        documents = []

        if not os.path.exists(data_path):
            logger.error(f"Data path {data_path} does not exist.")
            return []

        try:
            # ---------------------------
            # 📄 Load PDFs
            # ---------------------------
            pdf_loader = DirectoryLoader(
                data_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            documents.extend(pdf_loader.load())

            # ---------------------------
            # 📄 Load TXT files
            # ---------------------------
            txt_loader = DirectoryLoader(
                data_path,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents.extend(txt_loader.load())

            logger.info(f"Total raw documents loaded: {len(documents)}")

            if not documents:
                logger.warning("No documents found.")
                return []

            # ---------------------------
            # 🧹 STRICT CLEANING (FIXED)
            # ---------------------------
            cleaned_documents = []

            for doc in documents:
                try:
                    text = doc.page_content

                    # Strict validation
                    if not isinstance(text, str):
                        continue

                    text = text.strip()

                    # Remove empty / small / garbage text
                    if len(text) < 30:
                        continue

                    # Fix encoding issues
                    text = text.encode("utf-8", errors="ignore").decode("utf-8")

                    if not text or text.isspace():
                        continue

                    doc.page_content = text
                    cleaned_documents.append(doc)

                except Exception as e:
                    logger.debug(f"Skipping problematic document: {e}")

            logger.info(f"Valid documents after cleaning: {len(cleaned_documents)}")

            if not cleaned_documents:
                logger.error("No valid documents after cleaning!")
                return []

            # ---------------------------
            # ✂️ CHUNKING (SAFE)
            # ---------------------------
            if settings.USE_SEMANTIC_CHUNKING and self.semantic_splitter:
                logger.info("Using Semantic Chunking...")

                try:
                    semantic_chunks = self.semantic_splitter.split_documents(cleaned_documents)
                    logger.info(f"Semantic chunks created: {len(semantic_chunks)}")

                    # Hybrid refinement
                    logger.info("Refining with recursive splitting...")
                    chunks = self.recursive_splitter.split_documents(semantic_chunks)

                except Exception as e:
                    logger.error(f"Semantic chunking failed: {e}")
                    logger.warning("Falling back to recursive chunking...")

                    chunks = self.recursive_splitter.split_documents(cleaned_documents)

            else:
                logger.info("Using Recursive Character splitting...")
                chunks = self.recursive_splitter.split_documents(cleaned_documents)

            logger.info(f"Final chunks generated: {len(chunks)}")

            return chunks

        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []