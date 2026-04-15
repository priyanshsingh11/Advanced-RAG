import logging
import sys
import os

# Ensure we can import from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from app.services.document_loader import DocumentLoader
from app.db.qdrant_store import QdrantStore
from rich.console import Console
from rich.panel import Panel

# Setup logging
logging.basicConfig(level=logging.INFO)
console = Console()

def main():
    console.print(Panel.fit(
        "[bold green]Starting Document Ingestion...[/bold green]\n"
        "[dim]Processing PDFs from the data/ folder into Qdrant[/dim]",
        border_style="green"
    ))

    try:
        # 1. Initialize Loader and Store
        loader = DocumentLoader()
        store = QdrantStore()

        # 2. Load and Split Documents
        console.print("[yellow]Reading and splitting PDFs...[/yellow]")
        chunks = loader.load_and_split(data_path="./data")

        if not chunks:
            console.print("[bold red]No documents found in the data/ directory![/bold red]")
            return

        # 3. Upsert into Qdrant
        console.print(f"[yellow]Ingesting {len(chunks)} chunks into Qdrant storage...[/yellow]")
        console.print("[dim italic]Processing in batches to optimize RAM usage.[/dim italic]")
        success = store.upsert_documents(chunks, batch_size=500)

        if success:
            console.print(Panel(
                f"[bold green]Ingestion Complete![/bold green]\n"
                f"Successfully indexed {len(chunks)} chunks.\n"
                f"You can now run 'python cli_compare.py' to test the system.",
                border_style="bold green"
            ))
        else:
            console.print("[bold red]Upsert failed. Check the logs for errors.[/bold red]")

    except Exception as e:
        console.print(f"[bold red]Unexpected Error during ingestion:[/bold red] {str(e)}")

if __name__ == "__main__":
    main()
