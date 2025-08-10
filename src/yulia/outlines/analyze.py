import os
import pandas as pd
import numpy as np
from typing import Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def compute_model_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean and std dev of metrics for each model.
    """
    # Calculate means
    means = df.groupby("model").agg({
        "precision": ["mean", "std"],
        "recall": ["mean", "std"],
        "f1": ["mean", "std"],
        "conciseness": ["mean", "std"],
        "redundancy": ["mean", "std"]
    }).round(3)
    
    # Flatten column names
    means.columns = [f"{col[0]}_{col[1]}" for col in means.columns]
    return means.reset_index()

def plot_model_metrics(df: pd.DataFrame, 
                     metrics: list = ["precision", "recall", "f1", "conciseness", "redundancy"],
                     plot_type: str = "bar",
                     output_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
    """
    Create plots comparing models across different metrics.
    
    Args:
        df: DataFrame with model statistics
        metrics: List of metrics to plot
        plot_type: Type of plot ("bar" or "line")
        output_dir: Directory to save plots, if None returns dict of figures
    """
    figures = {}
    
    # Group metrics by type for better visualization
    performance_metrics = ["precision", "recall", "f1"]
    quality_metrics = ["conciseness", "redundancy"]
    
    # Create two subplot groups
    if any(m in metrics for m in performance_metrics):
        perf_metrics = [m for m in metrics if m in performance_metrics]
        fig_perf = create_metric_plot(df, perf_metrics, "Model Performance Metrics", plot_type)
        figures["performance"] = fig_perf
        
    if any(m in metrics for m in quality_metrics):
        qual_metrics = [m for m in metrics if m in quality_metrics]
        fig_qual = create_metric_plot(df, qual_metrics, "Outline Quality Metrics", plot_type)
        figures["quality"] = fig_qual
    
    # Save or return figures
    if output_dir:
        for name, fig in figures.items():
            out_path = Path(output_dir) / f"model_{name}_{plot_type}.png"
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
        return None
    return figures

def create_metric_plot(df: pd.DataFrame, metrics: list, title: str, plot_type: str = "bar") -> plt.Figure:
    """Helper function to create either bar or line plot for given metrics."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    
    if plot_type == "bar":
        width = 0.8 / len(metrics)  # Adjust bar width based on metric count
        for i, metric in enumerate(metrics):
            means = df[f"{metric}_mean"]
            stds = df[f"{metric}_std"]
            pos = x + (i - len(metrics)/2 + 0.5) * width
            ax.bar(pos, means, width, yerr=stds, capsize=5, label=metric.capitalize())
    else:  # line plot
        for metric in metrics:
            means = df[f"{metric}_mean"]
            stds = df[f"{metric}_std"]
            ax.plot(x, means, marker='o', label=metric.capitalize())
            ax.fill_between(x, means-stds, means+stds, alpha=0.2)
    
    # Customize plot
    ax.set_xlabel("Models")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add threshold lines if applicable
    from yulia.outlines.config import THRESHOLDS
    for metric in metrics:
        if metric in THRESHOLDS:
            threshold = THRESHOLDS[metric]
            if metric in ["conciseness", "redundancy"]:
                ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5,
                          label=f"{metric.capitalize()} Threshold")
            else:
                ax.axhline(y=threshold, color='g', linestyle='--', alpha=0.5,
                          label=f"{metric.capitalize()} Threshold")
    
    plt.tight_layout()
    return fig

def get_plots_dir(csv_path: str) -> str:
    """
    Create plots directory based on CSV name in the results directory
    """
    from yulia.outlines.config import RESULTS_DIR
    
    # Create plots directory in results dir
    plots_dir = os.path.join(RESULTS_DIR, "plots")
    
    # Get CSV name without extension for subdirectory
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    # Final path: results_dir/plots/csv_name/
    return os.path.join(plots_dir, csv_name)

def analyze_results(csv_path: str, make_plots: bool = False, plot_type: str = "bar", output_dir: Optional[str] = None):
    """
    Analyze results from the outline generation experiment.
    
    Args:
        csv_path: Path to the CSV file with results
        make_plots: Whether to generate plots
        plot_type: Type of plot ("bar" or "line")
        output_dir: Directory to save plots (if make_plots is True)
    """
    # Read results
    df = pd.read_csv(csv_path)
    
    # Compute statistics
    stats = compute_model_stats(df)
    
    # Print summary
    print("\n=== Model Performance Statistics ===")
    print("\nPerformance Metrics (mean ± std):")
    perf_cols = ["model"] + [c for c in stats.columns if any(m in c for m in ["precision", "recall", "f1"])]
    print(stats[perf_cols].to_string(index=False))
    
    print("\nQuality Metrics (mean ± std):")
    qual_cols = ["model"] + [c for c in stats.columns if any(m in c for m in ["conciseness", "redundancy"])]
    print(stats[qual_cols].to_string(index=False))
    
    # Generate plots if requested
    if make_plots:
        # Use provided output_dir or generate based on CSV name
        plots_dir = output_dir if output_dir else get_plots_dir(csv_path)
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_model_metrics(stats, plot_type=plot_type, output_dir=plots_dir)
        print(f"\nPlots saved to: {plots_dir}")
        print(f"Created both performance metrics and quality metrics plots")
    
    return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze outline generation results")
    parser.add_argument("csv_path", help="Path to results CSV file")
    parser.add_argument("--plots", action="store_true", help="Generate plots")
    parser.add_argument("--plot-type", choices=["bar", "line"], default="bar", help="Type of plot to generate")
    parser.add_argument("--output-dir", help="Directory to save plots")
    
    args = parser.parse_args()
    analyze_results(args.csv_path, args.plots, args.plot_type, args.output_dir)
