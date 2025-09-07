"""Generate embeddings for existing outlines."""

import os
import sys
import torch
import traceback
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import HfApi
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

# Set CUDA timeout and launch blocking
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
if torch.cuda.is_available():
    # Increase timeout to 30 seconds (default is usually around 2 seconds)
    torch.cuda.set_device(0)  # Explicitly set device
    torch.cuda.empty_cache()  # Clear cache before starting
    current_device = torch.cuda.current_device()
    torch.cuda._sleep(100)  # Small delay to ensure GPU is ready

from yulia.outlines.config import (
    RESULTS_DIR, HF_REPO_ID, VERSION
)

# Configuration
CHUNK_SIZE = 100000  # Size of embedding chunks to save

def init_sonar():
    """Initialize SONAR pipeline with GPU if available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use float32 for better stability, even on GPU
    dtype = torch.float32
    
    print(f"\nInitializing SONAR on {device} with {dtype}...")
    try:
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name()}")
            
        if device.type == "cuda":
            torch.cuda.empty_cache()
            
        text2vec = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
            dtype=dtype
        )
        return text2vec
    except Exception as e:
        print(f"Failed to initialize on {device}, error: {str(e)}")
        print("Traceback:", traceback.format_exc())
        print("Falling back to CPU")
        device = torch.device("cpu")
        dtype = torch.float32
        text2vec = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
            dtype=dtype
        )
        return text2vec

def generate_embeddings_for_chunk(
    outlines: list,
    text2vec: TextToEmbeddingModelPipeline,
    chunk_id: int,
    version: str,
    temp_dir: Path,
    api: HfApi,
    test_mode: bool = False
) -> None:
    """Generate and save embeddings for a chunk of outlines."""
    # Generate embeddings in batches with progress bar
    embeddings_list = []
    # Smaller batch size for GPU to avoid memory issues, larger for CPU
    batch_size = 64 if torch.cuda.is_available() else 512  # Larger batches for CPU to better utilize vectorization
    for i in tqdm(range(0, len(outlines), batch_size), desc=f"Chunk {chunk_id:03d}", unit="batch", leave=False):
        batch = outlines[i:i + batch_size]
        embedding = text2vec.predict(batch, source_lang="eng_Latn")
        embeddings_list.append(embedding)
    
    # Stack all embeddings
    embeddings = torch.cat(embeddings_list, dim=0)
    
    # Save embeddings locally (permanent storage)
    local_emb_dir = Path(RESULTS_DIR) / f"v{version}/embeddings"
    local_emb_dir.mkdir(parents=True, exist_ok=True)
    local_file = local_emb_dir / f"outlines_{chunk_id:03d}.pt"
    torch.save(embeddings, local_file)
    
    # Only upload to HuggingFace if not in test mode
    if not test_mode:
        # Save to temp for HuggingFace upload
        temp_file = temp_dir / f"outlines_{chunk_id:03d}.pt"
        torch.save(embeddings, temp_file)
        
        # Upload to HuggingFace
        remote_path = f"v{version}/embeddings/outlines_{chunk_id:03d}.pt"
        api.upload_file(
            path_or_fileobj=str(temp_file),
            path_in_repo=remote_path,
            repo_id=HF_REPO_ID,
            repo_type="dataset"
        )
        
        # Clean up temp file
        temp_file.unlink()

def main(
    version: str = "5.0",
    chunk_size: int = CHUNK_SIZE,
    single_chunk: int = None,
    test_mode: bool = False
):
    print(f"Starting embeddings generation for v{version}...")
    print(f"Config:")
    print(f"  Chunk size: {chunk_size}")
    
    text2vec = init_sonar()
    api = HfApi()
    
    # Create temporary directory for intermediate files
    temp_dir = Path(RESULTS_DIR) / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nProcessing outlines from HuggingFace...")
    
    try:
        if single_chunk is not None:
            # Process only the specified chunk
            chunk_id = single_chunk
            print(f"\nProcessing only chunk {chunk_id:03d}...")
            dataset = load_dataset(HF_REPO_ID, data_files=f"v{version}/data/outlines_{chunk_id:03d}.parquet", split="train")
            df = dataset.to_pandas()
            
            if df.empty:
                print(f"Warning: Chunk {chunk_id:03d} is empty")
            else:
                # Print first 5 example IDs
                print(f"\nFirst 5 example IDs in chunk {chunk_id:03d}:")
                print(df['example_id'].head().tolist())
                
                if test_mode:
                    print(f"\nTEST MODE: Processing first 10 samples from chunk {chunk_id:03d}...")
                    df = df.head(10)
                else:
                    print(f"\nProcessing chunk {chunk_id:03d} with {len(df)} samples...")
                outlines = df['outline_generated'].tolist()
                generate_embeddings_for_chunk(outlines, text2vec, chunk_id, version, temp_dir, api, test_mode=test_mode)
        else:
            # Process all chunks
            chunk_id = 0
            pbar = tqdm(desc="Processing chunks", unit="chunk")
            while True:
                try:
                    # Load the current chunk
                    dataset = load_dataset(HF_REPO_ID, data_files=f"v{version}/data/outlines_{chunk_id:03d}.parquet", split="train")
                    df = dataset.to_pandas()
                    
                    if df.empty:
                        break
                        
                    # Print first 5 example IDs
                    print(f"\nFirst 5 example IDs in chunk {chunk_id:03d}:")
                    print(df['example_id'].head().tolist())
                    
                    print(f"\nProcessing chunk {chunk_id:03d} with {len(df)} samples...")
                    
                    # Get outlines for this chunk
                    outlines = df['outline_generated'].tolist()
                    
                    # Generate and save embeddings
                    generate_embeddings_for_chunk(outlines, text2vec, chunk_id, version, temp_dir, api, test_mode=test_mode)
                    
                    chunk_id += 1
                    pbar.update(1)
                    
                except Exception as e:
                    if chunk_id == 0:
                        print(f"Error: No outline files found in v{version}/data/")
                        raise
                    print(f"\nFinished processing all chunks. Last chunk: {chunk_id-1}")
                    break
                    
    except Exception as e:
        print(f"Error processing chunk: {str(e)}")
        raise
    

    
    print("\nEmbeddings generation complete!")
    print(f"- Processed samples")
    print(f"- Uploaded to {HF_REPO_ID}/v{version}/embeddings/")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for existing outlines")
    parser.add_argument("--version", default="5.0", help="Version string for input/output files")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Number of samples per embedding chunk")
    parser.add_argument("--chunk", type=int, help="Process only this specific chunk number (e.g. 0 for chunk_000)")
    parser.add_argument("--test", action="store_true", help="Test mode: process only first 10 samples")
    
    args = parser.parse_args()
    
    try:
        main(
            version=args.version,
            chunk_size=args.chunk_size,
            single_chunk=args.chunk,
            test_mode=args.test
        )
    except Exception as e:
        print("Uncaught exception:", e, file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
