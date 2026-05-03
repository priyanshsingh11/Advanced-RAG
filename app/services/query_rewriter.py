from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings
import logging
import json

logger = logging.getLogger(__name__)

class QueryRewriter:
    def __init__(self):
        # Initializing the Groq model for fast query analysis
        self.llm = ChatGroq(
            model=settings.GROQ_MODEL,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert search assistant. Your task is to analyze the user query and output a JSON object.
            
            Fields:
            1. "rewritten_query": The user's query optimized for vector search (semantic) and BM25 search (keywords).
            2. "filters": A list of book or file names mentioned by the user to restrict the search. If none are mentioned, return an empty list [].
            
            Example output:
            {{
                "rewritten_query": "difference between bagging and boosting in ensemble learning",
                "filters": ["AI.pdf"]
            }}"""),
            ("user", "User Query: {query}")
        ])
        
        self.chain = self.prompt | self.llm

    def rewrite(self, query: str) -> dict:
        """Rewrites the user query and extracts filters."""
        try:
            logger.info(f"Analyzing query for rewriting and filters...")
            response = self.chain.invoke({"query": query})
            
            # Parse JSON response
            data = json.loads(response.content.strip())
            return {
                "rewritten_query": data.get("rewritten_query", query),
                "filters": data.get("filters", [])
            }
        except Exception as e:
            logger.error(f"Error in QueryRewriter: {e}")
            return {"rewritten_query": query, "filters": []}
