import os
import sys
import torch
import traceback
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from yulia.outlines.io_utils import load_sample, extract_outline_for_model
from yulia.outlines.config import (
    MODELS, HF_DATASET, HF_SPLIT, N_SAMPLES, RESULTS_DIR,
    SONAR_BATCH_SIZE, PROCESS_BATCH_SIZE,
    HF_REPO_ID, HF_PRIVATE, VERSION
)
from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16

def init_sonar(device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> Tuple[TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline]:
    """Initialize SONAR pipeline with optional device/dtype override"""
    if device is None:
        device = DEVICE
    if dtype is None:
        dtype = DTYPE
        
    print(f"Initializing SONAR pipeline on {device} with {dtype}")
    
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            
        text2vec = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
            dtype=dtype,
        )
        vec2text = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
            dtype=dtype,
        )
        return text2vec, vec2text
    except Exception as e:
        print(f"Failed to initialize on {device}, falling back to CPU: {e}")
        # Fall back to CPU
        device = torch.device("cpu")
        dtype = torch.float32
        text2vec = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
            dtype=dtype,
        )
        vec2text = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
            dtype=dtype,
        )
        return text2vec, vec2text

def generate_embeddings(
    text2vec: TextToEmbeddingModelPipeline,
    texts: List[str],
    batch_size: int = 32,
    show_progress: bool = True
) -> torch.Tensor:
    """Generate SONAR embeddings in batches with optional progress tracking"""
    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    
    # Use tqdm for progress if requested
    batch_iter = tqdm(range(0, len(texts), batch_size)) if show_progress else range(0, len(texts), batch_size)
    
    for i in batch_iter:
        if show_progress:
            batch_iter.set_description(f"Generating embeddings batch {i//batch_size + 1}/{n_batches}")
            
        batch = texts[i:i + batch_size]
        try:
            embeddings = text2vec.predict(batch, source_lang="eng_Latn")
            all_embeddings.append(embeddings)
        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            raise
            
    return torch.cat(all_embeddings, dim=0)

def process_batch(
    batch_samples: List[Dict],
    batch_id: int,
    text2vec: Optional[TextToEmbeddingModelPipeline],
    vec2text: Optional[EmbeddingToTextModelPipeline],
    run_dir: Path,
    version: str,
    with_reconstruction: bool = False,
    batch_size: int = 32,
    total_processed: int = 0
) -> Tuple[List[Dict], Optional[torch.Tensor]]:
    """Process a batch of samples, generating outlines and optionally embeddings"""
    rows = []
    outlines_for_embedding = []
    embedding_map = {}  # Maps outline position to row index
    
    for i, ex in enumerate(batch_samples):
        example_id = total_processed + i
        prompt = ex["prompt"]
        completion = ex["completion"]
        
        for model_idx, model in enumerate(MODELS):
            outline, used_prompt = extract_outline_for_model(model, completion)
            
            row_data = {
                "example_id": example_id,
                "model": model,
                "prompt": prompt,
                "completion": completion,
                "outline_prompt_used": used_prompt,
                "outline_generated": outline,
                "reconstructed_text": None  # Will be filled later if reconstruction is enabled
            }
            
            if text2vec is not None:
                # Store outline for batch embedding
                outline_pos = len(outlines_for_embedding)
                outlines_for_embedding.append(outline)
                embedding_map[outline_pos] = len(rows)  # Map position to row index
                
                # Add embedding_id that will map to the saved tensor
                row_data["embedding_id"] = total_processed * len(MODELS) + len(rows)
            
            rows.append(row_data)
    
    # Generate embeddings for the batch if needed
    embeddings = None
    if text2vec is not None and outlines_for_embedding:
        print(f"Generating embeddings for batch {batch_id}...")
        embeddings = generate_embeddings(text2vec, outlines_for_embedding, batch_size)
        
        # Generate reconstructed text if requested
        if with_reconstruction and vec2text is not None:
            print(f"Reconstructing text from embeddings...")
            reconstructed_texts = vec2text.predict(embeddings, target_lang="eng_Latn")
            
            # Add reconstructed text to rows
            for i, row_idx in enumerate(embedding_map.values()):
                rows[row_idx]["reconstructed_text"] = reconstructed_texts[i]
        
        # Save this batch of embeddings
        from yulia.outlines.hf_utils import save_embeddings_chunk
        chunk_path = save_embeddings_chunk(embeddings, batch_id, run_dir, version)
        print(f"Saved {len(outlines_for_embedding)} embeddings to {os.path.basename(chunk_path)}")
    
    return rows, embeddings

def generate_outlines(
    samples: List[Dict],
    run_dir: Path,
    version: str,
    with_embeddings: bool = False,
    with_reconstruction: bool = False,
    text2vec: Optional[TextToEmbeddingModelPipeline] = None,
    vec2text: Optional[EmbeddingToTextModelPipeline] = None,
    batch_size: int = 32,
    process_batch_size: int = 100  # Number of outlines to make in each batch
) -> pd.DataFrame:
    if with_embeddings and text2vec is None:
        raise ValueError("text2vec must be provided when with_embeddings=True")
    
    if with_reconstruction:
        if not with_embeddings:
            raise ValueError("--embeddings must be enabled to use --reconstruct")
        if vec2text is None:
            raise ValueError("vec2text must be provided when with_reconstruction=True")
    
    all_rows = []
    total_processed = 0
    
    # Process samples in batches
    for batch_id, batch_start in enumerate(range(0, len(samples), process_batch_size)):
        batch_end = min(batch_start + process_batch_size, len(samples))
        batch = samples[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_id + 1} ({batch_start + 1}-{batch_end} of {len(samples)})")
        
        # Process this batch
        batch_rows, _ = process_batch(
            batch_samples=batch,
            batch_id=batch_id,
            text2vec=text2vec if with_embeddings else None,
            vec2text=vec2text if with_embeddings else None,
            run_dir=run_dir,
            version=version,
            with_reconstruction=with_reconstruction,
            batch_size=batch_size,
            total_processed=total_processed
        )
        
        all_rows.extend(batch_rows)
        total_processed += len(batch)
        
        # Create a temporary DataFrame for this batch
        batch_df = pd.DataFrame(batch_rows)
        
        # Save intermediate results
        temp_csv = run_dir / f"outlines_temp_{batch_id:03d}.csv"
        batch_df.to_csv(temp_csv, index=False)
        print(f"Saved intermediate results to {temp_csv}")
    
    # Combine all batches into final DataFrame
    df = pd.DataFrame(all_rows).sort_values(["example_id", "model"]).reset_index(drop=True)
    
    # Clean up temporary files
    for temp_file in run_dir.glob("outlines_temp_*.csv"):
        temp_file.unlink()
    
    return df



def main(
    version: str = VERSION,
    with_embeddings: bool = False,
    with_reconstruction: bool = False,
    batch_size: int = SONAR_BATCH_SIZE,
    process_batch_size: int = PROCESS_BATCH_SIZE,
    save_to_hub: bool = False
):
    print("Starting outline pipeline...", flush=True)
    print(f"Config:", flush=True)
    print(f"  MODELS={MODELS}", flush=True)
    print(f"  DATASET={HF_DATASET}/{HF_SPLIT}", flush=True)
    print(f"  N_SAMPLES={N_SAMPLES}", flush=True)
    print(f"  BATCH_SIZE={batch_size}", flush=True)
    print(f"  PROCESS_BATCH_SIZE={process_batch_size}", flush=True)

    if not os.environ.get("DEEPINFRA_API_KEY"):
        print("DEEPINFRA_API_KEY is NOT set — API calls will fail.", flush=True)

    # Create run directory first
    from yulia.outlines.hf_utils import create_run_dir
    run_dir, _ = create_run_dir(Path(RESULTS_DIR), version)
    print(f"Created run directory: {run_dir}")

    # Initialize SONAR if needed
    text2vec = None
    vec2text = None
    if with_embeddings:
        print("Initializing SONAR...", flush=True)
        text2vec, vec2text = init_sonar()

    print("Loading samples…", flush=True)
    samples = load_sample(HF_DATASET, HF_SPLIT, N_SAMPLES)
    print(f"Loaded {len(samples)} samples", flush=True)

    print("Generating outlines" + (" and embeddings" if with_embeddings else "") + "…", flush=True)
    df = generate_outlines(
        samples=samples,
        run_dir=run_dir,
        version=version,
        with_embeddings=with_embeddings,
        with_reconstruction=with_reconstruction,
        text2vec=text2vec,
        vec2text=vec2text,
        batch_size=batch_size,
        process_batch_size=process_batch_size
    )
    
    print("Saving final results...", flush=True)
    
    if save_to_hub:
        from yulia.outlines.hf_utils import save_to_hub
        # Get all embeddings from saved chunks
        all_embeddings = []
        for chunk_file in sorted((run_dir / "sonar_embeds").glob("chunk_*.pt")):
            all_embeddings.append(torch.load(chunk_file))
        embeddings = torch.cat(all_embeddings, dim=0)
        
        # Save to HuggingFace
        hub_url = save_to_hub(
            df=df,
            embeddings=embeddings,
            repo_id=HF_REPO_ID,
            version=version,
            private=HF_PRIVATE
        )
        print(f"\nData uploaded to: {hub_url}")
    else:
        # Save locally
        df.to_csv(run_dir / "outlines.csv", index=False)
        print(f"\nResults saved in: {run_dir}")
        print(f"- outlines.csv: Main data ({len(df)} rows)")
        print(f"- sonar_embeds/: Embeddings saved in chunks")
    
    print(f"\nDataFrame shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate outlines with optional SONAR embeddings")
    parser.add_argument("--version", default=VERSION, help="Version string for output files")
    parser.add_argument("--embeddings", action="store_true", help="Generate SONAR embeddings for outlines")
    parser.add_argument("--reconstruct", action="store_true", help="Reconstruct text from SONAR embeddings")
    parser.add_argument("--batch-size", type=int, default=SONAR_BATCH_SIZE, help="Batch size for SONAR embedding generation")
    parser.add_argument("--process-batch-size", type=int, default=PROCESS_BATCH_SIZE, help="Number of samples to process in each batch")
    parser.add_argument("--save-to-hub", action="store_true", help="Save results directly to HuggingFace Hub")
    
    args = parser.parse_args()
    
    try:
        main(
            version=args.version,
            with_embeddings=args.embeddings,
            with_reconstruction=args.reconstruct,
            batch_size=args.batch_size,
            process_batch_size=args.process_batch_size,
            save_to_hub=args.save_to_hub
        )
    except Exception as e:
        print("Uncaught exception:", e, file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
