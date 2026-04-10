from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class QueryRewriter:
    def __init__(self):
        # Initializing the local Ollama model
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert search assistant. Your task is to rephrase the user's query to make it more suitable for a vector search engine and keyword search (BM25). Focus on extracting core entities and intents."),
            ("user", "Original Query: {query}\n\nRephrased Query:")
        ])
        
        self.chain = self.prompt | self.llm

    def rewrite(self, query: str) -> str:
        """Rewrites the user query for optimized retrieval."""
        try:
            logger.info(f"Rewriting query using {settings.OLLAMA_MODEL}")
            response = self.chain.invoke({"query": query})
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error in QueryRewriter: {e}")
            return query # Fallback to original query
