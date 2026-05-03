from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class HyDEGenerator:
    def __init__(self):
        self.llm = ChatGroq(
            model=settings.GROQ_MODEL,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0.2,
            model_kwargs={"top_p": 0.9}
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert academic researcher. Given a question, write a short, professional, and factual paragraph that answers the question. Do not include any citations or meta-talk, just the answer content."),
            ("user", "Question: {question}")
        ])
        
        self.chain = self.prompt | self.llm

    def generate_hypothetical_answer(self, query: str) -> str:
        """Generates a hypothetical answer to be used for semantic search."""
        try:
            logger.info(f"Generating HyDE hypothetical answer for: {query}")
            response = self.chain.invoke({"question": query})
            hyde_answer = response.content.strip()
            logger.debug(f"HyDE Answer: {hyde_answer[:100]}...")
            return hyde_answer
        except Exception as e:
            logger.error(f"Error generating HyDE answer: {e}")
            return query  # Fallback to original query
