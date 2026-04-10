from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class Generator:
    def __init__(self):
        # Initializing the local Ollama model
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.1
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional assistant for Advanced RAG. 
Answer the user's question ONLY based on the provided context. 
If the context doesn't contain the answer, say you don't know based on the context.
Always cite your sources clearly.

Context:
{context}"""),
            ("user", "{question}")
        ])
        
        self.chain = self.prompt | self.llm

    def generate(self, query: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generates an answer based on the provided context documents using local Ollama."""
        try:
            logger.info(f"Generating answer using {settings.OLLAMA_MODEL}")
            
            # 1. Prepare context string
            context_text = "\n\n".join([f"Source: {d['metadata'].get('source', 'Unknown')}\nContent: {d['content']}" for d in context_docs])
            
            # 2. Generate response
            response = self.chain.invoke({
                "context": context_text,
                "question": query
            })
            
            # 3. Estimate confidence score (Heuristic based on reranker scores)
            confidence = self._estimate_confidence(context_docs)
            
            return {
                "answer": response.content,
                "sources": list(set([d['metadata'].get('source', 'Unknown') for d in context_docs])),
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error in Generator: {e}")
            return {"answer": f"Unexpected error during generation: {str(e)}", "sources": [], "confidence": 0.0}

    def _estimate_confidence(self, docs: List[Dict[str, Any]]) -> float:
        """Heuristic confidence score based on reranker scores."""
        if not docs:
            return 0.0
        
        scores = [d.get("rerank_score", 0.5) for d in docs]
        avg_score = sum(scores) / len(scores)
        
        # Simple normalization for BGE reranker scores
        return min(max(avg_score, 0.0), 1.0)
