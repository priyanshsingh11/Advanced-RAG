from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings
from typing import List, Dict, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)

class Generator:
    def __init__(self):
        # Default LLM (compatible with existing code)
        self.default_llm = ChatOllama(
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
        
        self.default_chain = self.prompt | self.default_llm

    def generate(self, query: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generates an answer based on the provided context documents using default Ollama."""
        try:
            logger.info(f"Generating answer using default: {settings.OLLAMA_MODEL}")
            context_text = self._prepare_context(context_docs)
            
            response = self.default_chain.invoke({
                "context": context_text,
                "question": query
            })
            
            confidence = self._estimate_confidence(context_docs)
            
            return {
                "answer": response.content,
                "sources": list(set([d['metadata'].get('source', 'Unknown') for d in context_docs])),
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"Error in Generator: {e}")
            return {"answer": f"Unexpected error during generation: {str(e)}", "sources": [], "confidence": 0.0}

    def generate_with_benchmark(self, query: str, context_docs: List[Dict[str, Any]], model_name: str, provider: str) -> Dict[str, Any]:
        """Generates an answer and tracks performance metrics."""
        try:
            logger.info(f"Benchmarking {model_name} from {provider}")
            
            # 1. Initialize specific LLM
            if provider == "ollama":
                llm = ChatOllama(model=model_name, base_url=settings.OLLAMA_BASE_URL, temperature=0.1)
            elif provider == "groq":
                llm = ChatGroq(model=model_name, groq_api_key=settings.GROQ_API_KEY, temperature=0.1)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            chain = self.prompt | llm
            context_text = self._prepare_context(context_docs)

            # 2. Execute with timing
            start_time = time.time()
            response = chain.invoke({
                "context": context_text,
                "question": query
            })
            duration = time.time() - start_time

            # 3. Extract tokens
            usage = getattr(response, 'usage_metadata', {})
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)

            return {
                "model_name": model_name,
                "provider": provider,
                "answer": response.content,
                "time_taken": round(duration, 3),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "confidence": self._estimate_confidence(context_docs)
            }

        except Exception as e:
            logger.error(f"Error benchmarking {model_name}: {e}")
            return {
                "model_name": model_name,
                "provider": provider,
                "answer": f"Error: {str(e)}",
                "time_taken": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "confidence": 0.0
            }

    def _prepare_context(self, docs: List[Dict[str, Any]]) -> str:
        return "\n\n".join([f"Source: {d['metadata'].get('source', 'Unknown')}\nContent: {d['content']}" for d in docs])

    def _estimate_confidence(self, docs: List[Dict[str, Any]]) -> float:
        """Heuristic confidence score based on reranker scores."""
        if not docs:
            return 0.0
        scores = [d.get("rerank_score", 0.5) for d in docs]
        avg_score = sum(scores) / len(scores)
        return min(max(avg_score, 0.0), 1.0)
