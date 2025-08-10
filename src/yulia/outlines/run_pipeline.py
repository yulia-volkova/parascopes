import os
import sys
import traceback
import pandas as pd
from typing import List, Dict

from yulia.outlines.io_utils import load_sample, extract_outline_for_model
from yulia.outlines.metrics import semantic_prf1, conciseness_ratio, redundancy_score
from yulia.outlines.config import MODELS, HF_DATASET, HF_SPLIT, N_SAMPLES, GENERATIONS_CSV


def generate_outlines(samples: List[Dict]) -> pd.DataFrame:
    rows = []
    for example_id, ex in enumerate(samples):
        prompt = ex["prompt"]
        completion = ex["completion"]
        print(f"[{example_id+1}/{len(samples)}] example started", flush=True)

        for model in MODELS:
            print(f"  → model: {model}", flush=True)
            outline, used_prompt = extract_outline_for_model(model,completion)

            prf = semantic_prf1(completion, outline)
            conc = conciseness_ratio(completion, outline)
            red  = redundancy_score(outline)

            rows.append({
                "example_id": example_id,
                "model": model,
                "prompt": prompt,
                "completion": completion,
                "outline_prompt_used": used_prompt,
                "outline_generated": outline,
                "precision": prf["precision"],
                "recall": prf["recall"],
                "f1": prf["f1"],
                "conciseness": conc,
                "redundancy": red,
            })
        print(f"[{example_id+1}/{len(samples)}] example done", flush=True)
    df = pd.DataFrame(rows).sort_values(["example_id","model"]).reset_index(drop=True)
    return df

def main(analyze: bool = False, make_plots: bool = False, plots_dir: str = None, plot_type: str = "bar"):
    print("Starting outline pipeline...", flush=True)
    print(f"Config:\n  MODELS={MODELS}\n  DATASET={HF_DATASET}/{HF_SPLIT}\n  N_SAMPLES={N_SAMPLES}\n  OUT={GENERATIONS_CSV}", flush=True)

    if not os.environ.get("DEEPINFRA_API_KEY"):
        print("DEEPINFRA_API_KEY is NOT set — API calls will fail.", flush=True)

    print("Loading samples…", flush=True)
    samples = load_sample(HF_DATASET, HF_SPLIT, N_SAMPLES)
    print(f"Loaded {len(samples)} samples", flush=True)

    print("Generating outlines + metrics…", flush=True)
    df = generate_outlines(samples)

    print("💾 Saving CSV…", flush=True)
    df.to_csv(GENERATIONS_CSV, index=False)
    print(f"Saved → {GENERATIONS_CSV}", flush=True)

    # Basic summary
    summary = (
        df.groupby("model", as_index=False)
          .agg({"precision":"mean","recall":"mean","f1":"mean","conciseness":"mean","redundancy":"mean"})
          .sort_values(["f1","recall"], ascending=False)
    )
    print("\n=== Model Summary (means) ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    
    # Optional detailed analysis
    if analyze:
        from yulia.outlines.analyze import analyze_results
        print("\nRunning detailed analysis...")
        analyze_results(GENERATIONS_CSV, make_plots, plot_type, plots_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate and analyze outlines")
    parser.add_argument("--analyze", action="store_true", help="Run detailed analysis")
    parser.add_argument("--plots", action="store_true", help="Generate comparison plots")
    parser.add_argument("--plot-type", choices=["bar", "line"], default="bar", help="Type of plot to generate")
    parser.add_argument("--plots-dir", help="Directory to save plots")
    
    args = parser.parse_args()
    
    try:
        main(analyze=args.analyze, make_plots=args.plots, plots_dir=args.plots_dir, plot_type=args.plot_type)
    except Exception as e:
        print("Uncaught exception:", e, file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
