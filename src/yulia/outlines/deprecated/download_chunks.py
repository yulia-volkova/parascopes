"""Script to download chunks and save as CSV."""

import os
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi
from yulia.outlines.config import HF_REPO_ID

def download_chunk(version: str, chunk_id: str):
    api = HfApi()
    chunk_path = f"v{version}/data/outlines_{chunk_id}.parquet"
    local_path = chunk_path
    
    print(f"\nDownloading {chunk_path}...")
    api.hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=chunk_path,
        local_dir="."
    )
    
    print(f"Loading chunk_{chunk_id}...")
    dataset = load_dataset("parquet", data_files=local_path, split="train")
    df = dataset.to_pandas()
    
    print(f"\nChunk {chunk_id} statistics:")
    print(f"Total samples: {len(df)}")
    print(f"ID ranges:")
    print(f"- example_id: {df['example_id'].min()} to {df['example_id'].max()}")
    print(f"- dataset_idx: {df['dataset_idx'].min()} to {df['dataset_idx'].max()}")
    print(f"- embedding_id: {df['embedding_id'].min()} to {df['embedding_id'].max()}")
    
    # Save to CSV
    save_dir = Path("/workspace/ALGOVERSE/yas/yulia/parascopes/src/yulia/outlines/results") / "all_9_chunks"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"chunk_{chunk_id}.csv"
    
    print(f"\nSaving entire chunk to {save_path}")
    df.to_csv(save_path, index=False)
    print(f"✓ Saved {len(df)} samples")
    
    # Cleanup
    os.remove(local_path)

def main(version: str = "5.0"):
    chunk_ids = [f"{i:03d}" for i in range(9)]  # Creates ["000", "001", ..., "008"]
    
    print(f"Downloading {len(chunk_ids)} chunks from v{version}/data/")
    print(f"Chunk IDs: {chunk_ids}")
    print(f"Saving to: results/all_9_chunks/")
    
    for chunk_id in chunk_ids:
        try:
            download_chunk(version, chunk_id)
        except Exception as e:
            print(f"\nError downloading chunk_{chunk_id}: {e}")
    
    print(f"\nDone! Downloaded {len(chunk_ids)} chunks to results/all_9_chunks/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="5.0")
    args = parser.parse_args()
    main(args.version)





