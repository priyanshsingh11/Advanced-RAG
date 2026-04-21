from ragas import EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings # Ragas often defaults to this, but we can override
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings
from typing import List, Dict, Any, Optional
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

class RagasEvaluator:
    def __init__(self):
        # 1. Initialize Evaluator LLM (using Groq)
        if settings.GROQ_API_KEY:
            self.evaluator_llm = ChatGroq(
                model=settings.EVALUATOR_MODEL,
                groq_api_key=settings.GROQ_API_KEY,
                temperature=0
            )
        else:
            logger.warning("GROQ_API_KEY not found. Ragas evaluation might fail or need OpenAI.")
            self.evaluator_llm = None

        # 2. Initialize Embeddings for metrics that need them (like answer_relevancy)
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.DENSE_EMBEDDING_MODEL)

        # 3. Define metrics
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]

    def run_evaluation(self, 
                       questions: List[str], 
                       answers: List[str], 
                       contexts: List[List[str]], 
                       ground_truths: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Runs Ragas evaluation on the provided data.
        """
        if not self.evaluator_llm:
            raise ValueError("Evaluator LLM (Groq) is not initialized. Please check your API keys.")

        try:
            # Create the dataset format required by Ragas
            data = {
                "user_input": questions,
                "response": answers,
                "retrieved_contexts": contexts,
            }
            if ground_truths:
                data["reference"] = ground_truths

            dataset = EvaluationDataset.from_list([
                {
                    "user_input": q,
                    "response": a,
                    "retrieved_contexts": c,
                    "reference": gt if ground_truths else ""
                }
                for q, a, c, gt in zip(questions, answers, contexts, ground_truths if ground_truths else [""] * len(questions))
            ])

            logger.info("Starting Ragas evaluation...")
            # Run evaluation
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.evaluator_llm,
                embeddings=self.embeddings
            )

            # Convert to pandas for easier manipulation
            df = result.to_pandas()
            return df

        except Exception as e:
            logger.error(f"Error during Ragas evaluation: {e}")
            raise e
