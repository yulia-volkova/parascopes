import os
import sys
import torch
import traceback
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from yulia.outlines.io_utils import load_sample, extract_outline_for_model
from yulia.outlines.id_utils import initialize_id_counter
from yulia.outlines.config import (
    MODELS, HF_DATASET, HF_SPLIT, N_SAMPLES, RESULTS_DIR,
    SONAR_BATCH_SIZE, PROCESS_BATCH_SIZE,
    HF_REPO_ID, HF_PRIVATE, VERSION
)

# Firts version of outlines generation pipeline that was later replaced by the parallelized one

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

# here we work with samples of original dataset "nickypro/fineweb-llama3b-regen"
def process_batch(
    batch_samples: List[Dict],
    batch_id: int,
    text2vec: Optional[TextToEmbeddingModelPipeline],
    vec2text: Optional[EmbeddingToTextModelPipeline],
    version: str,
    with_reconstruction: bool = False,
    batch_size: int = 32,
    total_processed: int = 0,
    start_idx: Optional[int] = None,
    version_dir: Optional[Path] = None,
    next_id: Optional[int] = None  # Base ID to start from
) -> Tuple[List[Dict], Optional[torch.Tensor]]:
    """Process a batch of samples, generating outlines and optionally embeddings"""
    rows = []
    outlines_for_embedding = []
    embedding_map = {}  # Maps outline position to row index
    
    for i, ex in enumerate(batch_samples):
        print(f"Processing sample: index={i}, dataset_idx={ex['dataset_idx']}, start_idx={start_idx}")
        
        # Use the provided next_id as the base, incrementing by position in batch and total processed
        example_id = (next_id if next_id is not None else 0) + total_processed + i
        
        # dataset_idx is the original position in the dataset
        dataset_idx = ex["dataset_idx"]
        print(f"Processing: example_id={example_id} (our count), dataset_idx={dataset_idx} (original position)")
        
        completion = ex["completion"]
        
        for model_idx, model in enumerate(MODELS):
            # here we create outline using API call
            outline, used_prompt = extract_outline_for_model(model, completion)
            
            row_data = {
                "example_id": example_id,        # Our processing count (0, 1, 2, etc.)
                "dataset_idx": dataset_idx,      # Original position in dataset
                "model": model,
                "completion": completion,
                "outline_generated": outline,
                "reconstructed_text": ""         # Always string type, empty if no reconstruction
            }
            
            if text2vec is not None:
                # Store outline for batch embedding
                outline_pos = len(outlines_for_embedding)
                outlines_for_embedding.append(outline)
                embedding_map[outline_pos] = len(rows)  # Map position to row index
                
                # embedding_id should match example_id so we can map embeddings back to examples
                row_data["embedding_id"] = example_id
                print(f"  - Created row with example_id={example_id}, dataset_idx={dataset_idx}, embedding_id={example_id}")
            
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
                
        
    return rows, embeddings 

def generate_outlines(
    samples: List[Dict],
    version: str,
    with_embeddings: bool = False,
    with_reconstruction: bool = False,
    text2vec: Optional[TextToEmbeddingModelPipeline] = None,
    vec2text: Optional[EmbeddingToTextModelPipeline] = None,
    batch_size: int = 32,
    process_batch_size: int = 100,  # Number of outlines to make in each batch
    save_local: bool = True, 
    save_to_hub: bool = False,  
    version_dir: Optional[Path] = None,  # Directory for local files
    progress: Optional['ProgressTracker'] = None,  # Progress tracker instance
    start_chunk_id: Optional[int] = None,  # Starting chunk ID (if None, use progress tracker)
    items_per_chunk: int = 1000,  # Number of items to store in each chunk file
    start_idx: Optional[int] = None  # Starting index for processing
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
    
    # Initialize the ID counter once at the start
    next_id = initialize_id_counter(version_dir) if version_dir is not None else 0
    
    # Process samples in batches
    for batch_idx, batch_start in enumerate(range(0, len(samples), process_batch_size)):
        batch_end = min(batch_start + process_batch_size, len(samples))
        batch = samples[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_idx + 1} ({batch_start + 1}-{batch_end} of {len(samples)})")
        
        batch_rows, batch_embeddings = process_batch(
            batch_samples=batch,
            batch_id=batch_idx,
            text2vec=text2vec if with_embeddings else None,
            vec2text=vec2text if with_embeddings else None,
            version=version,
            with_reconstruction=with_reconstruction,
            batch_size=batch_size,
            total_processed=total_processed,
            start_idx=start_idx,
            version_dir=version_dir,
            next_id=next_id
        )
        
        all_rows.extend(batch_rows)
        total_processed += len(batch)
        
        # Create DataFrame for this batch
        batch_df = pd.DataFrame(batch_rows)
        
        # Save locally if requested
        if save_local:
            # Get the highest example_id in this batch
            highest_id = batch_df['example_id'].max()
            # Calculate chunk based on the highest ID (chunk_000 for 0-999, chunk_001 for 1000-1999, etc)
            current_chunk = highest_id // items_per_chunk
            
            data_dir = version_dir / "data"
            chunk_file = data_dir / f"chunk_{current_chunk:03d}.csv"
            
            # Check if we should append to existing chunk
            if chunk_file.exists():
                # Read existing chunk
                existing_df = pd.read_csv(chunk_file)
                # Get highest ID in existing chunk
                highest_existing_id = existing_df['example_id'].max()
                highest_batch_id = batch_df['example_id'].max()
                # Only append if both belong to the same thousand-range chunk
                if (highest_existing_id // items_per_chunk) == (highest_batch_id // items_per_chunk):
                    # Append new data
                    combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
                    # Save back
                    combined_df.to_csv(chunk_file, index=False)
                    print(f"Appended to chunk_{current_chunk:03d}.csv - now contains {len(combined_df)} items")
                else:
                    # Create new chunk file as we've crossed the thousand boundary
                    new_chunk = highest_batch_id // items_per_chunk
                    new_chunk_file = data_dir / f"chunk_{new_chunk:03d}.csv"
                    batch_df.to_csv(new_chunk_file, index=False)
                    print(f"Created new chunk_{new_chunk:03d}.csv with {len(batch_df)} items")
            else:
                # Create new chunk file
                batch_df.to_csv(chunk_file, index=False)
                print(f"Created new chunk_{current_chunk:03d}.csv with {len(batch_df)} items")
            
            # Print current chunk info
            if chunk_file.exists():
                existing_df = pd.read_csv(chunk_file)
                print(f"Current chunk has {existing_df['example_id'].nunique()} samples (example_ids: {existing_df['example_id'].unique()})")
            
            # Save embeddings if present
            if batch_embeddings is not None:
                emb_dir = version_dir / "embeddings"
                emb_dir.mkdir(exist_ok=True)
                emb_file = emb_dir / f"chunk_{current_chunk:03d}.pt"
                
                if emb_file.exists():
                    # Load and concatenate embeddings
                    existing_emb = torch.load(emb_file)
                    combined_emb = torch.cat([existing_emb, batch_embeddings], dim=0)
                    torch.save(combined_emb, emb_file)
                else:
                    torch.save(batch_embeddings, emb_file)
        
        # Upload to HF if requested
        if save_to_hub:
            from yulia.outlines.hf_utils import save_to_hub
            hub_url = save_to_hub(
                df=batch_df,
                embeddings=batch_embeddings,
                repo_id=HF_REPO_ID,
                version=version,
                chunk_id=current_chunk,  # Use the same chunk ID as local files
                private=HF_PRIVATE,
                items_per_chunk=items_per_chunk,
                with_reconstruction=with_reconstruction
            )
            print(f"Batch uploaded to: {hub_url}")
        
    
    # Create final df
    df = pd.DataFrame(all_rows).sort_values(["example_id", "model"]).reset_index(drop=True)
    
    # Update final progress
    if progress is not None:
        # Get total completed samples from DataFrame
        total_completed = df['example_id'].nunique()
        last_chunk = total_completed // items_per_chunk
        
        print(f"\nFinal progress update: {total_completed} samples completed")
        progress.update_progress(
            completed_samples=total_completed,
            chunk_idx=last_chunk,
            start_idx=start_idx
        )
    
    return df



def main(
    version: str = VERSION,
    with_embeddings: bool = False,
    with_reconstruction: bool = False,
    batch_size: int = SONAR_BATCH_SIZE,
    process_batch_size: int = PROCESS_BATCH_SIZE,
    save_to_hub: bool = False,
    save_local: bool = True,  # Whether to keep local copies
    start_idx: int = None,    # Start from this sample index
    n_samples: int = None,    # Process this many samples
    items_per_chunk: int = 1000  # Number of items to store in each chunk file
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

    # Create run directory and initialize progress tracking
    from yulia.outlines.progress import ProgressTracker
    
    version_dir = Path(RESULTS_DIR) / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    if save_local:
        (version_dir / "data").mkdir(exist_ok=True)
        if with_embeddings:
            (version_dir / "embeddings").mkdir(exist_ok=True)
    
    progress = ProgressTracker(version_dir)
    if start_idx is None:
        start_idx = progress.get_next_start_index()
    print(f"Starting from sample index: {start_idx}")
    print(f"Current progress: {progress.get_completion_status()}")

    text2vec = None
    vec2text = None
    if with_embeddings:
        print("Initializing SONAR...", flush=True)
        text2vec, vec2text = init_sonar()

    print("Loading samples…", flush=True)
    samples = load_sample(
        dataset_name=HF_DATASET,
        split=HF_SPLIT,
        n=n_samples,
        start_idx=start_idx if start_idx is not None else 0
    )
    
    # Update progress tracker with total samples we'll process in this run
    # If we're starting from an index, add it to get the true total
    total_to_process = len(samples)
    if start_idx is not None:
        total_to_process += start_idx
    progress.set_total_samples(total_to_process)
    print(f"Total samples to process (including previous): {total_to_process}")
    
    print(f"Processing dataset indices: {samples[0]['dataset_idx']} to {samples[-1]['dataset_idx']}")
    
    print(f"Loaded {len(samples)} samples to process", flush=True)

    print("Generating outlines" + (" and embeddings" if with_embeddings else "") + "…", flush=True)
    # Get starting chunk ID from progress tracker
    start_chunk_id = None
    if progress is not None:
        start_chunk_id = progress.get_next_chunk_index()
    
    df = generate_outlines(
        samples=samples,
        version=version,
        with_embeddings=with_embeddings,
        with_reconstruction=with_reconstruction,
        text2vec=text2vec,
        vec2text=vec2text,
        batch_size=batch_size,
        process_batch_size=process_batch_size,
        save_local=save_local,
        save_to_hub=save_to_hub,
        version_dir=version_dir,
        progress=progress,
        start_chunk_id=start_chunk_id,
        items_per_chunk=items_per_chunk
    )
    
    print("Saving final results...", flush=True)
    
    print(f"\nProcessing complete!")
    print(f"- Total samples processed: {len(df)}")
    print(f"- Version: {version}")
    
    if save_local:
        print(f"\nLocal files saved in: {version_dir}")
        print(f"- data/: CSV files with outlines")
        if with_embeddings:
            print(f"- embeddings/: PyTorch tensor files")
    
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
    parser.add_argument("--no-local", action="store_true", help="Don't save files locally")
    parser.add_argument("--start-idx", type=int, help="Start processing from this sample index")
    parser.add_argument("--n-samples", type=int, help="Process this many samples")
    parser.add_argument("--items-per-chunk", type=int, default=1000, help="Number of items to store in each chunk file")
    
    args = parser.parse_args()
    
    try:
        main(
            version=args.version,
            with_embeddings=args.embeddings,
            with_reconstruction=args.reconstruct,
            batch_size=args.batch_size,
            process_batch_size=args.process_batch_size,
            save_to_hub=args.save_to_hub,
            save_local=not args.no_local,
            start_idx=args.start_idx,
            n_samples=args.n_samples,
            items_per_chunk=args.items_per_chunk
        )
    except Exception as e:
        print("Uncaught exception:", e, file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
