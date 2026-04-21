import os
import sys
import pandas as pd
from rich.console import Console
from rich.table import Table
from app.services.orchestrator import RAGOrchestrator
from app.services.evaluator import RagasEvaluator
from dotenv import load_dotenv

# Initialize
load_dotenv()
console = Console()
orchestrator = RAGOrchestrator()
evaluator = RagasEvaluator()

def run_benchmark():
    # 1. Define Evaluation Set (This would ideally move to a JSON/CSV file)
    eval_set = [
        {
            "question": "What is the primary chunking strategy used in this project?",
            "ground_truth": "The project uses a hybrid strategy combining Semantic Chunking and Recursive Character Splitting."
        },
        {
            "question": "Which vector database is used for storing embeddings?",
            "ground_truth": "The project uses Qdrant as the vector database for storing and searching embeddings."
        },
        {
            "question": "What role does the cross-encoder play in the retrieval pipeline?",
            "ground_truth": "The cross-encoder is used for reranking the top retrieved documents to improve precision before generation."
        }
    ]

    questions = [item["question"] for item in eval_set]
    ground_truths = [item["ground_truth"] for item in eval_set]

    # 2. Collect RAG Results
    console.print("[bold blue]Collecting RAG answers and contexts...[/bold blue]")
    answers = []
    contexts = []

    for q in questions:
        # Run the full RAG query
        # Note: We need the raw results from the orchestrator
        # We need to reach into the internal retrieved docs before they are formatted for the LLM
        # But RAGOrchestrator.query returns a dict with answer and sources.
        # We need to modify query logic slightly or use a more direct path to get context strings.
        
        # Let's use a version of query that returns what RAGAS needs
        # For now, we'll simulate the call logic
        try:
            # Re-running logic to get contexts as List[str]
            rewritten = orchestrator.rewriter.rewrite(q)
            docs = orchestrator.retriever.retrieve(rewritten, top_k=5)
            reranked = orchestrator.reranker.rerank(rewritten, docs, top_k=3)
            
            # Context strings
            context_strings = [d["content"] for d in reranked]
            
            # Generate answer
            gen_result = orchestrator.generator.generate(q, reranked)
            
            answers.append(gen_result["answer"])
            contexts.append(context_strings)
            console.print(f"✓ Processed Q: {q[:50]}...")
        except Exception as e:
            console.print(f"[red]Error processing {q}: {e}[/red]")
            answers.append("ERROR")
            contexts.append([])

    # 3. Run RAGAS Evaluation
    console.print("\n[bold green]Starting RAGAS Scoring (this may take a minute)...[/bold green]")
    try:
        results_df = evaluator.run_evaluation(questions, answers, contexts, ground_truths)
        
        # 4. Display Results
        table = Table(title="RAGAS Evaluation Results")
        table.add_column("Question", style="cyan", no_wrap=False)
        table.add_column("Faithfulness", style="magenta")
        table.add_column("Relevancy", style="magenta")
        table.add_column("Precision", style="magenta")
        table.add_column("Recall", style="magenta")

        for _, row in results_df.iterrows():
            table.add_row(
                str(row["user_input"])[:50] + "...",
                f"{row['faithfulness']:.3f}",
                f"{row['answer_relevancy']:.3f}",
                f"{row['context_precision']:.3f}",
                f"{row['context_recall']:.3f}"
            )
        
        console.print(table)
        
        # Save to CSV
        output_file = "ragas_eval_results.csv"
        results_df.to_csv(output_file, index=False)
        console.print(f"\n[bold green]✓ Full results saved to {output_file}[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Evaluation failed: {e}[/bold red]")

if __name__ == "__main__":
    if not os.path.exists("./data"):
        console.print("[yellow]Warning: /data directory not found. Please ensure documents are indexed.[/yellow]")
    run_benchmark()
