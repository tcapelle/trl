from dataclasses import dataclass, field
from typing import List, Optional
import simple_parsing as sp
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


@dataclass
class Args:
    """Arguments for the dataset viewer script."""
    
    dataset_name: str = field(
        default="tcapelle/cuda-optimized-models",
        metadata={"help": "Name of the HuggingFace dataset to load"}
    )
    
    split: str = field(
        default="train",
        metadata={"help": "Dataset split to use"}
    )
    
    num_rows: int = field(
        default=10,
        metadata={"help": "Number of rows to display"}
    )
    
    columns: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Specific columns to display (displays all if None)"}
    )


def main():
    # Parse arguments
    args = sp.parse(Args)
    
    # Initialize Rich console
    console = Console()
    
    # Load the dataset
    console.print(Panel.fit(f"Loading dataset: [bold]{args.dataset_name}[/bold]", 
                            title="Dataset Viewer", 
                            border_style="green"))
    
    try:
        dataset = load_dataset(args.dataset_name, split=args.split)
        console.print(f"Total examples: [bold]{len(dataset)}[/bold]")
        
        # Get columns to display
        all_columns = list(dataset.features.keys())
        if args.columns:
            columns_to_show = [col for col in args.columns if col in all_columns]
            if not columns_to_show:
                console.print("[yellow]Warning: None of the specified columns exist in the dataset. Showing all columns.[/yellow]")
                columns_to_show = all_columns
        else:
            columns_to_show = all_columns
        
        # Create a table for dataset preview
        table = Table(title=f"Dataset Preview ({args.split} split)")
        
        # Add columns to the table
        for col in columns_to_show:
            table.add_column(col, overflow="fold")
        
        # Add rows to the table
        num_samples = min(args.num_rows, len(dataset))
        for i in range(num_samples):
            row_values = []
            for col in columns_to_show:
                value = dataset[i][col]
                # Handle different data types appropriately
                if isinstance(value, (list, dict)):
                    display_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                elif isinstance(value, str):
                    display_value = value[:100] + "..." if len(value) > 100 else value
                else:
                    display_value = str(value)
                row_values.append(display_value)
            table.add_row(*row_values)
        
        # Display the table
        console.print(table)
        
        # Display column information
        console.print(Panel.fit("Dataset Columns Information", border_style="blue"))
        column_table = Table(show_header=True)
        column_table.add_column("Column")
        column_table.add_column("Type")
        
        for col in all_columns:
            col_type = str(dataset.features[col])
            column_table.add_row(col, col_type)
        
        console.print(column_table)
        
    except Exception as e:
        console.print(f"[bold red]Error loading dataset:[/bold red] {str(e)}")


if __name__ == "__main__":
    main() 