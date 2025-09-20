"""Generate embeddings for outlines in chunks of 1000."""
import os
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import HfApi
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

# Config
SOURCE_REPO_ID = "yulia-volkova/parascopes-outlines-data-gemma27b"  # Gemma outlines
HF_REPO_ID = "yulia-volkova/gemma27b-outlines-embeddings"          # Gemma embeddings
RESULTS_DIR = Path(__file__).parent / "results"

def upload_to_hf(local_dir: Path):
    """Upload all generated embeddings to HuggingFace."""
    print("\nUploading embeddings to HuggingFace...")
    api = HfApi()
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)
    
    files = sorted(local_dir.glob("outlines_*.pt"))
    print(f"Found {len(files)} files to upload")
    
    for local_file in tqdm(files, desc="Uploading to HF"):
        chunk_id = int(local_file.stem.split("_")[1])
        api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=f"outlines_{chunk_id:03d}.pt",
            repo_id=HF_REPO_ID,
            repo_type="dataset"
        )
        print(f"Uploaded {local_file.name}")

def main(source_chunk_id: int, max_chunks: int = None):
    # Initialize SONAR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text2vec = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=device,
        dtype=torch.float32
    )
    
    # Load source chunk (100k outlines)
    print(f"\nLoading source chunk {source_chunk_id:03d}...")
    dataset = load_dataset(SOURCE_REPO_ID, data_files=f"v0.0/data/outlines_{source_chunk_id:03d}.parquet", split="train")
    df = dataset.to_pandas()
    outlines = df['outline_generated'].tolist()
    example_ids = df['example_id'].tolist()
    print(f"Loaded {len(outlines):,} outlines")
    print(f"ID range: {example_ids[0]} to {example_ids[-1]}")
    
    # Process in chunks of 1000
    chunk_size = 1000
    n_chunks = len(outlines) // chunk_size
    print(f"\nGenerating {n_chunks} embedding chunks of {chunk_size:,} samples each")
    
    # Create output directory
    local_dir = RESULTS_DIR / "gemma27b-outlines-embeddings"
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Process chunks
    batch_size = 64 if torch.cuda.is_available() else 512
    chunks_to_process = min(n_chunks, max_chunks) if max_chunks else n_chunks
    print(f"Will process {chunks_to_process} chunks")
    
    generated_files = []
    for chunk in tqdm(range(chunks_to_process), desc="Processing chunks", unit="chunk"):
        start_idx = chunk * chunk_size
        end_idx = start_idx + chunk_size
        chunk_id = source_chunk_id * 100 + chunk
        
        # Skip if file already exists
        local_file = local_dir / f"outlines_{chunk_id:03d}.pt"
        if local_file.exists():
            print(f"\nSkipping chunk {chunk_id:03d} - file already exists")
            generated_files.append(local_file)
            continue
        
        print(f"\nProcessing chunk {chunk_id:03d}")
        print(f"Samples {start_idx} to {end_idx-1}")
        print(f"Example IDs: {example_ids[start_idx]} to {example_ids[end_idx-1]}")
        
        # Generate embeddings in batches
        chunk_embeddings = []
        with tqdm(total=chunk_size, desc=f"Chunk {chunk_id:03d}", unit="sample", leave=False) as pbar:
            for i in range(start_idx, end_idx, batch_size):
                batch = outlines[i:min(i + batch_size, end_idx)]
                embedding = text2vec.predict(batch, source_lang="eng_Latn")
                chunk_embeddings.append(embedding)
                pbar.update(len(batch))
        
        # Concatenate and save
        embeddings = torch.cat(chunk_embeddings, dim=0)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Save locally
        torch.save(embeddings, local_file)
        generated_files.append(local_file)
        print(f"Saved to {local_file}")
        
        # Cleanup
        del embeddings
        del chunk_embeddings
        torch.cuda.empty_cache()
    
    print(f"\nGenerated {len(generated_files)} embedding files:")
    for f in generated_files:
        print(f"- {f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("chunk", type=int, help="Source chunk ID to process (e.g., 0 for outlines_000.parquet)")
    parser.add_argument("--max-chunks", type=int, help="Process only this many chunks (for testing)")
    parser.add_argument("--upload", action="store_true", help="Upload generated embeddings to HuggingFace")
    args = parser.parse_args()
    
    # Generate embeddings
    main(args.chunk, args.max_chunks)
    
    # Upload if requested
    if args.upload:
        local_dir = RESULTS_DIR / "gemma27b-outlines-embeddings"
        upload_to_hf(local_dir)