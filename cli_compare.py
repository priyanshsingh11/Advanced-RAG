import sys
import os
import time
from typing import List, Dict, Any

# Ensure we can import from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from app.services.orchestrator import RAGOrchestrator
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def display_welcome():
    console.print(Panel.fit(
        "[bold cyan]Advanced RAG - Model Benchmarking Tool[/bold cyan]\n"
        "[dim]Compare multiple LLMs side-by-side in your terminal[/dim]",
        border_style="blue"
    ))

def run_comparison(query: str):
    orchestrator = RAGOrchestrator()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Processing retrieval & reranking...", total=None)
        # The orchestrator.compare method runs the pipeline and benchmarks models
        results_data = orchestrator.compare(query)
    
    if "metadata" in results_data and "error" in results_data["metadata"]:
        console.print(f"[bold red]Error:[/bold red] {results_data['metadata']['error']}")
        return

    # Results Table
    table = Table(title=f"\nComparison for: [italic]\"{query}\"[/italic]", show_header=True, header_style="bold magenta", border_style="dim")
    table.add_column("Model", style="cyan", width=20)
    table.add_column("Time (s)", justify="right", style="green")
    table.add_column("Tokens (I/O)", justify="right", style="yellow")
    table.add_column("Conf.", justify="right")
    table.add_column("Answer Snippet", overflow="fold")

    for res in results_data["results"]:
        snippet = res["answer"][:150].replace("\n", " ") + "..."
        tokens = f"{res['input_tokens']}/{res['output_tokens']}"
        conf = f"{res['confidence']:.2f}"
        
        table.add_row(
            f"{res['model_name']} ({res['provider']})",
            str(res["time_taken"]),
            tokens,
            conf,
            snippet
        )

    console.print(table)
    
    # Detail View of each answer
    for idx, res in enumerate(results_data["results"]):
        console.print(Panel(
            res["answer"],
            title=f"[bold cyan]{res['model_name']} ({res['provider']})[/bold cyan]",
            subtitle=f"Time: {res['time_taken']}s | Total Tokens: {res['total_tokens']}",
            border_style="green" if idx == 0 else "blue",
            padding=(1, 2)
        ))

if __name__ == "__main__":
    display_welcome()
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = console.input("[bold yellow]Enter your query:[/bold yellow] ")
        
    if query.strip():
        try:
            run_comparison(query)
        except Exception as e:
            console.print(f"[bold red]Pipeline Error:[/bold red] {str(e)}")
    else:
        console.print("[dim]No query provided. Exiting.[/dim]")
