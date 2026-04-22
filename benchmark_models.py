"""
Model Benchmarking Script for Advanced RAG
==========================================
Runs 35 curated questions across all configured LLMs,
evaluates Accuracy & Faithfulness via LLM-as-judge,
and produces a scored CSV for model selection.

Final Score = (Accuracy * 0.5) + (Faithfulness * 0.3) + (Speed * 0.2)

Usage: python benchmark_models.py
"""

import sys
import os
import csv
import time
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from app.services.orchestrator import RAGOrchestrator
from app.core.config import settings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

console = Console()
logging.basicConfig(level=logging.WARNING)  # Suppress verbose logs during benchmarking

# ─────────────────────────────────────────────────────────
# 35 Curated Benchmark Questions (from your 4 ML/AI books)
# ─────────────────────────────────────────────────────────
BENCHMARK_QUESTIONS = [
    # ── Fundamentals ──
    {
        "question": "What is supervised learning and how does it differ from unsupervised learning?",
        "ground_truth": "Supervised learning uses labeled data to learn a mapping from inputs to outputs, while unsupervised learning finds patterns in unlabeled data without predefined output labels."
    },
    {
        "question": "What is overfitting in machine learning and how can it be prevented?",
        "ground_truth": "Overfitting occurs when a model learns noise in the training data and performs poorly on unseen data. It can be prevented using techniques like regularization, cross-validation, early stopping, and using more training data."
    },
    {
        "question": "Explain the bias-variance tradeoff in machine learning.",
        "ground_truth": "The bias-variance tradeoff refers to the balance between a model's ability to fit training data (low bias) and generalize to new data (low variance). High bias leads to underfitting, while high variance leads to overfitting."
    },
    {
        "question": "What is cross-validation and why is it used?",
        "ground_truth": "Cross-validation is a technique for evaluating model performance by partitioning data into training and validation sets multiple times. It provides a more robust estimate of model performance than a single train-test split."
    },
    {
        "question": "What is feature scaling and why is it important?",
        "ground_truth": "Feature scaling normalizes the range of features so they contribute equally to the model. Common methods include standardization (zero mean, unit variance) and min-max normalization. It is important for algorithms sensitive to feature magnitudes like SVM, KNN, and gradient descent."
    },

    # ── Classical ML ──
    {
        "question": "How does a decision tree algorithm work for classification?",
        "ground_truth": "A decision tree splits the data recursively based on feature values that maximize information gain or minimize impurity (Gini or entropy), creating a tree structure where leaves represent class labels."
    },
    {
        "question": "What is a random forest and how does it improve over a single decision tree?",
        "ground_truth": "A random forest is an ensemble of decision trees trained on random subsets of data and features. It reduces overfitting and improves generalization by averaging predictions across multiple trees."
    },
    {
        "question": "Explain how support vector machines (SVM) work.",
        "ground_truth": "SVMs find the optimal hyperplane that maximizes the margin between classes. They use support vectors (data points closest to the boundary) and can handle non-linear boundaries using kernel functions."
    },
    {
        "question": "What is the K-Nearest Neighbors (KNN) algorithm?",
        "ground_truth": "KNN classifies a data point based on the majority class of its K nearest neighbors in the feature space. It is a lazy learning algorithm that does not build an explicit model during training."
    },
    {
        "question": "What is gradient descent and how does it optimize a model?",
        "ground_truth": "Gradient descent is an optimization algorithm that iteratively updates model parameters by moving in the direction of the negative gradient of the loss function, minimizing the error between predictions and actual values."
    },

    # ── Deep Learning ──
    {
        "question": "What is a neural network and what are its basic components?",
        "ground_truth": "A neural network consists of layers of interconnected neurons (nodes). It has an input layer, one or more hidden layers, and an output layer. Each connection has a weight, and neurons apply activation functions to their inputs."
    },
    {
        "question": "Explain the backpropagation algorithm in neural networks.",
        "ground_truth": "Backpropagation computes gradients of the loss function with respect to each weight by applying the chain rule of calculus, propagating errors backward through the network to update weights during training."
    },
    {
        "question": "What is the vanishing gradient problem in deep learning?",
        "ground_truth": "The vanishing gradient problem occurs when gradients become extremely small during backpropagation in deep networks, causing earlier layers to learn very slowly. It is commonly addressed using ReLU activation, batch normalization, and residual connections."
    },
    {
        "question": "How do convolutional neural networks (CNNs) process images?",
        "ground_truth": "CNNs use convolutional layers with learnable filters to detect spatial features like edges and textures. Pooling layers reduce spatial dimensions, and fully connected layers perform final classification."
    },
    {
        "question": "What are recurrent neural networks (RNNs) and what are they used for?",
        "ground_truth": "RNNs are neural networks designed for sequential data where the output depends on previous computations. They maintain a hidden state that captures information from prior time steps, commonly used for text, speech, and time series."
    },

    # ── Regularization & Optimization ──
    {
        "question": "What is the difference between L1 and L2 regularization?",
        "ground_truth": "L1 regularization (Lasso) adds the sum of absolute weights to the loss, encouraging sparsity. L2 regularization (Ridge) adds the sum of squared weights, shrinking weights towards zero but not exactly zero."
    },
    {
        "question": "What is dropout in neural networks?",
        "ground_truth": "Dropout is a regularization technique that randomly deactivates a fraction of neurons during training to prevent co-adaptation and reduce overfitting. At inference time, all neurons are active."
    },
    {
        "question": "What is batch normalization and why is it used?",
        "ground_truth": "Batch normalization normalizes the inputs of each layer to have zero mean and unit variance within a mini-batch. It stabilizes training, allows higher learning rates, and acts as a mild regularizer."
    },

    # ── Ensemble & Advanced ──
    {
        "question": "What is boosting and how does it differ from bagging?",
        "ground_truth": "Boosting trains models sequentially, each correcting errors of the previous one, producing a weighted combination. Bagging trains models independently on random subsets and aggregates via voting/averaging. Boosting reduces bias while bagging reduces variance."
    },
    {
        "question": "Explain how the gradient boosting algorithm works.",
        "ground_truth": "Gradient boosting builds trees sequentially where each new tree fits the residual errors (negative gradients) of the combined ensemble. It minimizes a loss function by adding weak learners in a gradient descent-like fashion."
    },

    # ── Dimensionality & Clustering ──
    {
        "question": "What is Principal Component Analysis (PCA)?",
        "ground_truth": "PCA is a dimensionality reduction technique that transforms features into a new set of orthogonal components ordered by the amount of variance they explain. It projects data onto the directions of maximum variance."
    },
    {
        "question": "How does the K-Means clustering algorithm work?",
        "ground_truth": "K-Means partitions data into K clusters by iteratively assigning points to the nearest centroid and updating centroids as the mean of assigned points until convergence."
    },

    # ── Evaluation ──
    {
        "question": "What is the difference between precision and recall?",
        "ground_truth": "Precision is the fraction of true positives among predicted positives (how many predicted positives are correct). Recall is the fraction of true positives among actual positives (how many actual positives were found)."
    },
    {
        "question": "What is the ROC curve and AUC score?",
        "ground_truth": "The ROC curve plots the true positive rate against the false positive rate at different classification thresholds. AUC (Area Under the Curve) summarizes the overall ability of the model to discriminate between classes."
    },

    # ── Practical ──
    {
        "question": "What is transfer learning and when is it useful?",
        "ground_truth": "Transfer learning reuses a model trained on one task as the starting point for a different but related task. It is useful when the target task has limited labeled data, leveraging knowledge learned from a larger source dataset."
    },

    # ── AI: A Modern Approach (Russell & Norvig) ──
    {
        "question": "What is the A* search algorithm and why is it optimal?",
        "ground_truth": "A* is a best-first search algorithm that uses f(n) = g(n) + h(n), where g(n) is the cost to reach node n and h(n) is a heuristic estimate to the goal. It is optimal when the heuristic is admissible (never overestimates) and consistent."
    },
    {
        "question": "What is a constraint satisfaction problem (CSP) and how is it solved?",
        "ground_truth": "A CSP consists of variables, domains, and constraints. It is solved by assigning values to variables such that all constraints are satisfied. Techniques include backtracking search, arc consistency, and constraint propagation."
    },
    {
        "question": "What are Bayesian networks and how do they represent probabilistic relationships?",
        "ground_truth": "Bayesian networks are directed acyclic graphs where nodes represent random variables and edges represent conditional dependencies. Each node has a conditional probability table that quantifies the effect of parents on the node."
    },
    {
        "question": "What is a Markov Decision Process (MDP)?",
        "ground_truth": "An MDP is a mathematical framework for sequential decision making under uncertainty. It consists of states, actions, transition probabilities, and rewards. The goal is to find an optimal policy that maximizes expected cumulative reward."
    },
    {
        "question": "What is the difference between informed and uninformed search strategies?",
        "ground_truth": "Uninformed search strategies like BFS and DFS have no knowledge about the goal location beyond the problem definition. Informed strategies like A* and greedy best-first use heuristic functions to estimate distance to the goal, making them more efficient."
    },
    {
        "question": "What is natural language processing and what are its main challenges?",
        "ground_truth": "Natural language processing deals with computational understanding and generation of human language. Main challenges include ambiguity, context dependence, syntax vs semantics, coreference resolution, and the vast variability of human expression."
    },
    {
        "question": "What is reinforcement learning and how does it differ from supervised learning?",
        "ground_truth": "Reinforcement learning is learning through interaction with an environment by receiving rewards or penalties. Unlike supervised learning which uses labeled examples, RL learns from trial and error to maximize cumulative reward over time."
    },
    {
        "question": "What is the minimax algorithm used for in game-playing AI?",
        "ground_truth": "Minimax is a decision-making algorithm for two-player zero-sum games. It recursively evaluates game states assuming the opponent plays optimally — maximizing the minimum payoff. Alpha-beta pruning optimizes it by eliminating branches that cannot affect the final decision."
    },
    {
        "question": "What is knowledge representation in artificial intelligence?",
        "ground_truth": "Knowledge representation is the field concerned with how to formally encode information about the world so that an AI system can use it for reasoning. Common approaches include propositional logic, first-order logic, semantic networks, and ontologies."
    },
    {
        "question": "What is the Turing Test and what does it measure?",
        "ground_truth": "The Turing Test, proposed by Alan Turing, evaluates a machine's ability to exhibit intelligent behavior indistinguishable from a human. A machine passes if a human evaluator cannot reliably distinguish the machine's responses from those of a human."
    },
]


# ─────────────────────────────────────────────────────
# LLM-as-Judge: Evaluate Accuracy & Faithfulness
# ─────────────────────────────────────────────────────

class LLMJudge:
    """Uses a local Ollama model to score Accuracy and Faithfulness (0-10)."""

    def __init__(self):
        self.llm = ChatOllama(
            model=settings.EVALUATOR_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0
        )

        self.eval_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a strict evaluator for a RAG (Retrieval-Augmented Generation) system.

You will be given:
- A QUESTION
- A GROUND TRUTH answer (the expected correct answer)
- A MODEL ANSWER (the answer to evaluate)

Score the MODEL ANSWER on two criteria. Each score is 0 to 10:

1. **Accuracy** (0-10): How factually correct is the model answer compared to the ground truth? 
   - 10 = perfectly matches ground truth
   - 5 = partially correct, missing key details
   - 0 = completely wrong or says "I don't know"

2. **Faithfulness** (0-10): Does the answer ONLY contain information that could be derived from the context/sources? Does it hallucinate or add unsupported claims?
   - 10 = fully faithful, no hallucination
   - 5 = mostly faithful with minor unsupported additions
   - 0 = heavily hallucinated

RESPOND ONLY with valid JSON in this exact format, nothing else:
{{"accuracy": <number>, "faithfulness": <number>}}"""),
            ("user", """QUESTION: {question}

GROUND TRUTH: {ground_truth}

MODEL ANSWER: {model_answer}

JSON scores:""")
        ])

        self.chain = self.eval_prompt | self.llm

    def evaluate(self, question: str, ground_truth: str, model_answer: str) -> Dict[str, float]:
        """Returns accuracy and faithfulness scores (0-10)."""
        try:
            response = self.chain.invoke({
                "question": question,
                "ground_truth": ground_truth,
                "model_answer": model_answer
            })

            # Parse JSON from response
            content = response.content.strip()
            # Handle cases where model wraps JSON in markdown blocks
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            scores = json.loads(content)
            return {
                "accuracy": min(max(float(scores.get("accuracy", 0)), 0), 10),
                "faithfulness": min(max(float(scores.get("faithfulness", 0)), 0), 10)
            }
        except Exception as e:
            # If parsing fails, return neutral scores
            logging.warning(f"Judge parsing error: {e} | Raw: {response.content if 'response' in dir() else 'N/A'}")
            return {"accuracy": 5.0, "faithfulness": 5.0}


# ─────────────────────────────────────────────────────
# Main Benchmarking Logic
# ─────────────────────────────────────────────────────

def run_benchmark():
    console.print(Panel.fit(
        "[bold cyan]Advanced RAG - Full Model Benchmark[/bold cyan]\n"
        "[dim]Running 35 questions across all models with LLM-as-judge scoring[/dim]",
        border_style="blue"
    ))

    # Initialize components
    console.print("[yellow]Initializing pipeline components...[/yellow]")
    orchestrator = RAGOrchestrator()
    judge = LLMJudge()

    # Get model list
    ollama_models = [m.strip() for m in settings.OLLAMA_MODELS.split(",")]
    models = [(m, "ollama") for m in ollama_models]
    if settings.GROQ_API_KEY:
        models.append((settings.GROQ_MODEL, "groq"))

    total_questions = len(BENCHMARK_QUESTIONS)
    total_models = len(models)
    total_runs = total_questions * total_models

    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Questions: {total_questions}")
    console.print(f"  Models: {total_models} → {[m[0] for m in models]}")
    console.print(f"  Total evaluations: {total_runs}")
    console.print(f"  Judge model: {settings.EVALUATOR_MODEL}\n")

    # Results storage
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        console=console,
    ) as progress:

        main_task = progress.add_task("[cyan]Overall progress", total=total_runs)

        for q_idx, q_item in enumerate(BENCHMARK_QUESTIONS, 1):
            question = q_item["question"]
            ground_truth = q_item["ground_truth"]

            # ── Step 1: Retrieval & Reranking (shared across models) ──
            progress.update(main_task, description=f"[cyan]Q{q_idx}/{total_questions}: Retrieving...")
            try:
                rewritten_query = orchestrator.rewriter.rewrite(question)
                retrieved_docs = orchestrator.retriever.retrieve(
                    rewritten_query, top_k=settings.TOP_K_RETRIEVAL
                )

                if not retrieved_docs:
                    # No context found — mark all models for this question
                    for model_name, provider in models:
                        all_results.append({
                            "Question": question,
                            "Model": f"{model_name} ({provider})",
                            "Answer": "No context retrieved",
                            "Accuracy": 0.0,
                            "Faithfulness": 0.0,
                            "Time_s": 0.0,
                            "Input_Tokens": 0,
                            "Output_Tokens": 0,
                            "Total_Tokens": 0,
                            "Speed_Score": 0.0,
                            "Final_Score": 0.0
                        })
                        progress.advance(main_task)
                    continue

                reranked_docs = orchestrator.reranker.rerank(
                    rewritten_query, retrieved_docs, top_k=settings.TOP_K_RERANK
                )
            except Exception as e:
                console.print(f"[red]Retrieval error for Q{q_idx}: {e}[/red]")
                for model_name, provider in models:
                    all_results.append({
                        "Question": question,
                        "Model": f"{model_name} ({provider})",
                        "Answer": f"Retrieval Error: {e}",
                        "Accuracy": 0.0,
                        "Faithfulness": 0.0,
                        "Time_s": 0.0,
                        "Input_Tokens": 0,
                        "Output_Tokens": 0,
                        "Total_Tokens": 0,
                        "Speed_Score": 0.0,
                        "Final_Score": 0.0
                    })
                    progress.advance(main_task)
                continue

            # ── Step 2: Generate with each model & evaluate ──
            for model_name, provider in models:
                progress.update(
                    main_task,
                    description=f"[cyan]Q{q_idx}/{total_questions} → {model_name}"
                )

                try:
                    # Generate with benchmarking
                    gen_result = orchestrator.generator.generate_with_benchmark(
                        question, reranked_docs, model_name, provider
                    )

                    answer = gen_result["answer"]
                    time_taken = gen_result["time_taken"]
                    input_tokens = gen_result["input_tokens"]
                    output_tokens = gen_result["output_tokens"]
                    total_tokens = gen_result["total_tokens"]

                    # Evaluate with LLM judge
                    scores = judge.evaluate(question, ground_truth, answer)
                    accuracy = scores["accuracy"]
                    faithfulness = scores["faithfulness"]

                    # Normalize accuracy and faithfulness to 0-1
                    norm_accuracy = accuracy / 10.0
                    norm_faithfulness = faithfulness / 10.0

                    # Speed score: normalize time (faster = higher score)
                    # Cap at 120s as worst case, 0.5s as best case
                    speed_score = max(0.0, min(1.0, 1.0 - (time_taken - 0.5) / 119.5))

                    # Final weighted score
                    final_score = (norm_accuracy * 0.5) + (norm_faithfulness * 0.3) + (speed_score * 0.2)

                    all_results.append({
                        "Question": question,
                        "Model": f"{model_name} ({provider})",
                        "Answer": answer[:500],  # Truncate for CSV readability
                        "Accuracy": round(accuracy, 2),
                        "Faithfulness": round(faithfulness, 2),
                        "Time_s": round(time_taken, 3),
                        "Input_Tokens": input_tokens,
                        "Output_Tokens": output_tokens,
                        "Total_Tokens": total_tokens,
                        "Speed_Score": round(speed_score, 4),
                        "Final_Score": round(final_score, 4)
                    })

                except Exception as e:
                    console.print(f"[red]  Error with {model_name}: {e}[/red]")
                    all_results.append({
                        "Question": question,
                        "Model": f"{model_name} ({provider})",
                        "Answer": f"Error: {e}",
                        "Accuracy": 0.0,
                        "Faithfulness": 0.0,
                        "Time_s": 0.0,
                        "Input_Tokens": 0,
                        "Output_Tokens": 0,
                        "Total_Tokens": 0,
                        "Speed_Score": 0.0,
                        "Final_Score": 0.0
                    })

                progress.advance(main_task)

    # ─────────────────────────────────────────────────
    # Save Detailed CSV
    # ─────────────────────────────────────────────────
    detail_csv = f"benchmark_results_{timestamp}.csv"
    fieldnames = [
        "Question", "Model", "Accuracy", "Faithfulness",
        "Time_s", "Input_Tokens", "Output_Tokens", "Total_Tokens",
        "Speed_Score", "Final_Score", "Answer"
    ]

    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    console.print(f"\n[bold green]✓ Detailed results saved to {detail_csv}[/bold green]")

    # ─────────────────────────────────────────────────
    # Aggregate Summary per Model
    # ─────────────────────────────────────────────────
    model_scores: Dict[str, Dict[str, Any]] = {}
    for row in all_results:
        model = row["Model"]
        if model not in model_scores:
            model_scores[model] = {
                "accuracy_sum": 0, "faithfulness_sum": 0, "speed_sum": 0,
                "final_sum": 0, "time_sum": 0, "tokens_sum": 0, "count": 0
            }
        model_scores[model]["accuracy_sum"] += row["Accuracy"]
        model_scores[model]["faithfulness_sum"] += row["Faithfulness"]
        model_scores[model]["speed_sum"] += row["Speed_Score"]
        model_scores[model]["final_sum"] += row["Final_Score"]
        model_scores[model]["time_sum"] += row["Time_s"]
        model_scores[model]["tokens_sum"] += row["Total_Tokens"]
        model_scores[model]["count"] += 1

    # Summary CSV
    summary_csv = f"benchmark_summary_{timestamp}.csv"
    summary_rows = []
    for model, data in model_scores.items():
        n = data["count"]
        summary_rows.append({
            "Model": model,
            "Avg_Accuracy": round(data["accuracy_sum"] / n, 2),
            "Avg_Faithfulness": round(data["faithfulness_sum"] / n, 2),
            "Avg_Speed_Score": round(data["speed_sum"] / n, 4),
            "Avg_Final_Score": round(data["final_sum"] / n, 4),
            "Avg_Time_s": round(data["time_sum"] / n, 2),
            "Total_Tokens": data["tokens_sum"],
            "Questions_Answered": n
        })

    # Sort by final score
    summary_rows.sort(key=lambda x: x["Avg_Final_Score"], reverse=True)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    console.print(f"[bold green]✓ Summary results saved to {summary_csv}[/bold green]\n")

    # ─────────────────────────────────────────────────
    # Display Summary Table
    # ─────────────────────────────────────────────────
    table = Table(
        title="[bold]Model Benchmark Summary[/bold]",
        show_header=True,
        header_style="bold magenta",
        border_style="dim"
    )
    table.add_column("Rank", style="bold", width=5)
    table.add_column("Model", style="cyan", width=35)
    table.add_column("Avg Accuracy\n(0-10)", justify="right", style="green")
    table.add_column("Avg Faithful\n(0-10)", justify="right", style="yellow")
    table.add_column("Avg Speed\n(0-1)", justify="right", style="blue")
    table.add_column("Avg Time (s)", justify="right")
    table.add_column("Final Score\n(weighted)", justify="right", style="bold white")

    for rank, row in enumerate(summary_rows, 1):
        style = "bold green" if rank == 1 else ""
        table.add_row(
            f"#{rank}",
            row["Model"],
            str(row["Avg_Accuracy"]),
            str(row["Avg_Faithfulness"]),
            str(row["Avg_Speed_Score"]),
            str(row["Avg_Time_s"]),
            str(row["Avg_Final_Score"]),
            style=style
        )

    console.print(table)

    # Winner
    winner = summary_rows[0]
    console.print(Panel.fit(
        f"[bold green]🏆 Best Model: {winner['Model']}[/bold green]\n"
        f"Final Score: {winner['Avg_Final_Score']} | "
        f"Accuracy: {winner['Avg_Accuracy']}/10 | "
        f"Faithfulness: {winner['Avg_Faithfulness']}/10 | "
        f"Avg Time: {winner['Avg_Time_s']}s",
        border_style="green",
        title="[bold]Winner[/bold]"
    ))


if __name__ == "__main__":
    start = time.time()
    run_benchmark()
    elapsed = time.time() - start
    console.print(f"\n[dim]Total benchmark time: {elapsed/60:.1f} minutes[/dim]")
