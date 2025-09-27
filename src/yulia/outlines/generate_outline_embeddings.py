"""Generate embeddings for existing outlines."""

import os
import sys
import torch
import traceback
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import HfApi
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

# Set CUDA timeout and launch blocking
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
if torch.cuda.is_available():
  
    torch.cuda.set_device(0)  
    torch.cuda.empty_cache()  #
    current_device = torch.cuda.current_device()
    torch.cuda._sleep(100) 

from yulia.outlines.config import (
    RESULTS_DIR, VERSION
)

# HuggingFace configuration
SOURCE_REPO_ID = "yulia-volkova/parascopes-outlines-data"  
HF_REPO_ID = "yulia-volkova/llama-3b-outlines-embeddings"  

# Configuration
CHUNK_SIZE = 1000  # Size of embedding chunks to save (1000 chunks with 1000 items each)
NUM_CHUNKS = 1000  # Total number of chunks to process

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
    example_ids: list,
    test_mode: bool = False
) -> None:
    # Generate embeddings in batches with progress bar
    embeddings_list = []
    # Smaller batch size for GPU to avoid memory issues, larger for CPU
    batch_size = 64 if torch.cuda.is_available() else 512
    
    # Process in sub-chunks of 1000 samples
    sub_chunk_size = 1000
    n_sub_chunks = (len(outlines) + sub_chunk_size - 1) // sub_chunk_size
    
    for sub_chunk in range(n_sub_chunks):
        start_idx = sub_chunk * sub_chunk_size
        end_idx = min(start_idx + sub_chunk_size, len(outlines))
        sub_chunk_id = chunk_id * 100 + sub_chunk  # e.g., chunk 0 -> sub_chunks 000-099
        
        print(f"\nProcessing sub-chunk {sub_chunk_id:03d}")
        print(f"Example ID range: {example_ids[start_idx]} to {example_ids[end_idx-1]}")
        
        sub_chunk_embeddings = []
        with tqdm(total=end_idx-start_idx, desc=f"Sub-chunk {sub_chunk_id:03d}", unit="sample", leave=False, position=0) as pbar:
            for i in range(start_idx, end_idx, batch_size):
                batch = outlines[i:i + batch_size]
                embedding = text2vec.predict(batch, source_lang="eng_Latn")
                sub_chunk_embeddings.append(embedding)
                pbar.update(len(batch))
        
        # Stack and save this sub-chunk
        sub_embeddings = torch.cat(sub_chunk_embeddings, dim=0)
        
        # Save embeddings locally (permanent storage)
        local_emb_dir = Path(RESULTS_DIR) / "llama-3b-outlines-embeddings"
        local_emb_dir.mkdir(parents=True, exist_ok=True)
        local_file = local_emb_dir / f"outlines_{sub_chunk_id:03d}.pt"
        torch.save(sub_embeddings, local_file)
        
        # Only upload to HuggingFace if not in test mode
        if not test_mode:
            temp_file = temp_dir / f"outlines_{sub_chunk_id:03d}.pt"
            torch.save(sub_embeddings, temp_file)
            
            remote_path = f"outlines_{sub_chunk_id:03d}.pt"
            api.upload_file(
                path_or_fileobj=str(temp_file),
                path_in_repo=remote_path,
                repo_id=HF_REPO_ID,
                repo_type="dataset"
            )
            
            # Clean up temp file
            temp_file.unlink()
            
        # Clear memory
        del sub_embeddings
        del sub_chunk_embeddings
        torch.cuda.empty_cache()

def main(
    version: str = "5.0",
    chunk_size: int = CHUNK_SIZE,
    single_chunk: int = None,
    test_mode: bool = False,
    start_chunk: int = 0,
    end_chunk: int = NUM_CHUNKS - 1
):
    print(f"Starting embeddings generation for v{version}...")
    print(f"Config:")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Processing chunks: {start_chunk:03d} to {end_chunk:03d}")
    
    text2vec = init_sonar()
    api = HfApi()
    
    try:
        api.create_repo(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            private=True,
            exist_ok=True
        )
        print(f"\nEnsured HuggingFace repository exists: {HF_REPO_ID}")
    except Exception as e:
        print(f"Note: Repository exists or error occurred: {e}")
    
    # Create temporary directory for intermediate files
    temp_dir = Path(RESULTS_DIR) / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nProcessing outlines from HuggingFace...")
    
    try:
        if single_chunk is not None:
            # Process only the specified chunk
            chunk_id = single_chunk
            if chunk_id >= NUM_CHUNKS:
                raise ValueError(f"Chunk ID {chunk_id} is too large. Max chunk ID is {NUM_CHUNKS-1}")
                
            print(f"\nProcessing only chunk {chunk_id:03d}...")
            dataset = load_dataset(SOURCE_REPO_ID, data_files=f"v{version}/data/outlines_{chunk_id:03d}.parquet", split="train")
            df = dataset.to_pandas()
            
            if df.empty:
                print(f"Warning: Chunk {chunk_id:03d} is empty")
            else:
                print(f"\nChunk {chunk_id:03d} info:")
                print(f"Total samples: {len(df)}")
                print(f"First 5 example IDs: {df['example_id'].head().tolist()}")
                print(f"Last 5 example IDs: {df['example_id'].tail().tolist()}")
                
                if test_mode:
                    print(f"\nTEST MODE: Processing first 10 samples...")
                    df = df.head(10)
                outlines = df['outline_generated'].tolist()
                example_ids = df['example_id'].tolist()
                generate_embeddings_for_chunk(outlines, text2vec, chunk_id, version, temp_dir, api, example_ids, test_mode=test_mode)
        else:
            # Process chunks in the specified range
            for chunk_id in range(start_chunk, end_chunk + 1):
                try:
                    print(f"\nProcessing chunk {chunk_id:03d} ({chunk_id-start_chunk+1}/{end_chunk-start_chunk+1})...")
                    dataset = load_dataset(SOURCE_REPO_ID, data_files=f"v{version}/data/outlines_{chunk_id:03d}.parquet", split="train")
                    df = dataset.to_pandas()
                    
                    if df.empty:
                        print(f"Warning: Chunk {chunk_id:03d} is empty, skipping...")
                        continue
                        
                    print(f"\nChunk {chunk_id:03d} info:")
                    print(f"Total samples: {len(df)}")
                    print(f"First 5 example IDs: {df['example_id'].head().tolist()}")
                    print(f"Last 5 example IDs: {df['example_id'].tail().tolist()}")
                    
                    # Get outlines and example IDs for this chunk
                    outlines = df['outline_generated'].tolist()
                    example_ids = df['example_id'].tolist()
                    
                    # Generate and save embeddings
                    generate_embeddings_for_chunk(outlines, text2vec, chunk_id, version, temp_dir, api, example_ids, test_mode=test_mode)
                    
                except Exception as e:
                    print(f"Error processing chunk {chunk_id:03d}: {str(e)}")
                    if chunk_id == start_chunk:
                        raise
                    print("Continuing with next chunk...")
                    continue
                    
    except Exception as e:
        print(f"Error processing chunk: {str(e)}")
        raise
    

    
    print("\nEmbeddings generation complete!")
    print(f"- Read outlines from {SOURCE_REPO_ID}/v{version}/data/")
    print(f"- Generated and saved embeddings to {HF_REPO_ID}/")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for existing outlines")
    parser.add_argument("--version", default="5.0", help="Version string for input/output files")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Number of samples per embedding chunk")
    parser.add_argument("--chunk", type=int, help="Process only this specific chunk number (e.g. 0 for chunk_000)")
    parser.add_argument("--start-chunk", type=int, default=0, help="Start processing from this chunk number")
    parser.add_argument("--end-chunk", type=int, default=NUM_CHUNKS-1, help="Process chunks up to this number (inclusive)")
    parser.add_argument("--test", action="store_true", help="Test mode: process only first 10 samples")
    
    args = parser.parse_args()
    
    try:
        main(
            version=args.version,
            chunk_size=args.chunk_size,
            single_chunk=args.chunk,
            test_mode=args.test,
            start_chunk=args.start_chunk,
            end_chunk=args.end_chunk
        )
    except Exception as e:
        print("Uncaught exception:", e, file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
