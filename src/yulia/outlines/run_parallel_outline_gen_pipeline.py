import os
import sys
import traceback
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import itertools
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

from yulia.outlines.io_utils import extract_outline_for_model
from yulia.outlines.config import (
    MODELS, HF_DATASET, HF_SPLIT,
    HF_REPO_ID, HF_PRIVATE, VERSION
)
from yulia.outlines.hf_utils import get_standard_features


NUM_WORKERS = 50  
BATCH_SIZE = 200 

def process_sample(sample: Dict[str, Any], example_id: int) -> Dict[str, Any]:
    completion = sample["completion"]
    model = MODELS[0]  
    
    outline, _ = extract_outline_for_model(model, completion)
    
    return {
        "example_id": example_id,
        "dataset_idx": sample["id"],  # Using 'id' from dataset as dataset_idx
        "model": model,
        "completion": completion,
        "outline_generated": outline,
        "reconstructed_text": "",  # Empty as we'll generate this later with embeddings if we choose to 
        "embedding_id": example_id  
    }

def process_batch(batch: List[Dict], start_id: int, batch_num: int = 0) -> List[Dict]:
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for i, sample in enumerate(batch):
            example_id = start_id + i
            # Print first 5 samples' IDs only for batch 0
            if batch_num == 0 and i < 5:
                print(f"\nProcessing sample {example_id}:")
                print(f"example_id:   {example_id}")
                print(f"dataset_idx:  {sample['id']}")
                print(f"embedding_id: {example_id}")
            futures.append(executor.submit(process_sample, sample, example_id))
        
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing sample: {e}")
                traceback.print_exc()
    
    return results

def upload_to_hub(df: pd.DataFrame, repo_id: str, version: str) -> None:
    api = HfApi()
    CHUNK_SIZE = 100000  # 100k samples per parquet
    
    # local results dir
    local_dir = Path(os.path.dirname(__file__)) / "results" / "gemma27b" / f"v{version}"
    local_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=HF_PRIVATE, exist_ok=True)
    except Exception as e:
        print(f"Note: Repository exists or error occurred: {e}")
    
    # Sort the dataframe by example_id to ensure proper ordering
    df = df.sort_values('example_id').reset_index(drop=True)
    
    # Find the current chunk based on the first example_id
    min_id = df['example_id'].min()
    max_id = df['example_id'].max()
    current_chunk = min_id // CHUNK_SIZE
    print(f"\nProcessing IDs {min_id}-{max_id} (chunks {min_id//CHUNK_SIZE:03d}-{max_id//CHUNK_SIZE:03d})")
    
    while current_chunk <= max_id // CHUNK_SIZE:
        chunk_start_id = current_chunk * CHUNK_SIZE
        # Special handling for final chunk (chunk 9) to allow up to 100,001 items althought I dont think this is needed
        if current_chunk == 9:
            chunk_end_id = min(max_id, 1000000)  # Allow up to 1,000,000 
        else:
            chunk_end_id = (current_chunk + 1) * CHUNK_SIZE - 1
        
        print(f"\nProcessing chunk_{current_chunk:03d} (IDs {chunk_start_id}-{chunk_end_id})")
        
        # Try to load existing chunk
        try:
            api.hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=f"v{version}/data/outlines_{current_chunk:03d}.parquet",
                local_dir="."
            )
            dataset = load_dataset(
                "parquet",
                data_files=f"v{version}/data/outlines_{current_chunk:03d}.parquet",
                split="train"
            )
            existing_df = dataset.to_pandas()
            print(f"Found existing chunk with {len(existing_df)} samples")
            print(f"ID range: {existing_df['example_id'].min()}-{existing_df['example_id'].max()}")
            # Clean up downloaded file
            os.remove(f"v{version}/data/outlines_{current_chunk:03d}.parquet")
        except Exception as e:
            print(f"No existing chunk found, will create new one: {e}")
            existing_df = None
            
        # Get new data for this chunk
        chunk_mask = (df['example_id'] >= chunk_start_id) & (df['example_id'] <= chunk_end_id)
        new_chunk_df = df[chunk_mask].copy()
        
        if len(new_chunk_df) > 0:
            print(f"Adding {len(new_chunk_df)} new samples")
            
            # Combine with existing data if any
            if existing_df is not None:
                # Remove any overlapping IDs from existing data
                existing_df = existing_df[~existing_df['example_id'].isin(new_chunk_df['example_id'])]
                chunk_df = pd.concat([existing_df, new_chunk_df], ignore_index=True)
                chunk_df = chunk_df.sort_values('example_id').reset_index(drop=True)
            else:
                chunk_df = new_chunk_df
            
            print(f"Final chunk size: {len(chunk_df)} samples")
            print(f"ID range: {chunk_df['example_id'].min()}-{chunk_df['example_id'].max()}")
            
            # Drop the index column and convert to HF dataset
            if '__index_level_0__' in chunk_df.columns:
                chunk_df = chunk_df.drop('__index_level_0__', axis=1)
            hf_dataset = Dataset.from_pandas(chunk_df, features=get_standard_features())
            
            
            hf_data_path = f"v{version}/data/outlines_{current_chunk:03d}.parquet"
            local_data_path = local_dir / f"outlines_{current_chunk:03d}.parquet"
            
            
            os.makedirs(os.path.dirname(hf_data_path), exist_ok=True)  # For HF upload
            hf_dataset.to_parquet(hf_data_path)
            
            # Also save to local results
            hf_dataset.to_parquet(str(local_data_path))
            print(f"✓ Saved locally to {local_data_path}")
            
            # Upload to HF
            api.upload_file(
                path_or_fileobj=hf_data_path,
                path_in_repo=hf_data_path,
                repo_id=repo_id,
                repo_type="dataset"
            )
            print(f"✓ Uploaded chunk_{current_chunk:03d} to HF")
            
            os.remove(hf_data_path)
        
        current_chunk += 1



def main(
    start_idx: int = 0,
    end_idx: int = 3,
    version: str = VERSION
):
    print(f"Starting parallel outline generation pipeline...")
    print(f"Config:")
    print(f"  DATASET={HF_DATASET}/{HF_SPLIT}")
    print(f"  Processing range: {start_idx}-{end_idx}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Workers: {NUM_WORKERS}")
    


    dataset = load_dataset(HF_DATASET, split=HF_SPLIT, streaming=True)
    
    print("\nDataset structure:")
    sample = next(iter(dataset))
    print("Available fields:", list(sample.keys()))
    
    # Process in batches
    current_id = start_idx
    batch = []
    results_buffer = []  
    CHUNK_SIZE = 100000  # 100k samples per chunk
    total_processed = 0
    
    print("\nGenerating outlines...")
    batch_num = 0
    print(f"\nStarting batch {batch_num}...")
    
    for data in tqdm(itertools.islice(dataset, start_idx, end_idx + 1), 
                    total=end_idx - start_idx + 1):
        batch.append(data)
        
        if len(batch) >= BATCH_SIZE:
            results = process_batch(batch, current_id, batch_num)
            results_buffer.extend(results)
            current_id += len(batch)
            total_processed += len(batch)
            batch = []
            batch_num += 1
            
            # Upload when collected 100k samples
            if len(results_buffer) >= CHUNK_SIZE:
                print(f"\nUploading chunk with {len(results_buffer)} samples...")
                df = pd.DataFrame(results_buffer).reset_index(drop=True)
                upload_to_hub(df, HF_REPO_ID, version)
                results_buffer = []  # Reset buffer
                print(f"Total processed so far: {total_processed}")
            else:
                # Print progress every 10k samples
                if len(results_buffer) % 10000 == 0:
                    current_chunk = current_id // CHUNK_SIZE
                    chunk_start = current_chunk * CHUNK_SIZE
                    chunk_end = (current_chunk + 1) * CHUNK_SIZE - 1
                    print(f"\nCollected {len(results_buffer)} samples in buffer")
                    print(f"Current chunk {current_chunk:03d} (ID range {chunk_start}-{chunk_end})")
                    print(f"Total processed so far: {total_processed}")
    
    # Process remaining items in the last batch
    if batch:
        results = process_batch(batch, current_id, batch_num)
        results_buffer.extend(results)
        total_processed += len(batch)
    
    # Upload any remaining results
    if results_buffer:
        print(f"\nUploading final batch of {len(results_buffer)} samples...")
        df = pd.DataFrame(results_buffer).reset_index(drop=True)
        upload_to_hub(df, HF_REPO_ID, version)
    
    print("\nProcessing complete!")
    print(f"- Total samples processed: {total_processed}")
    print(f"- ID range: {start_idx}-{current_id - 1}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate outlines in parallel")
    parser.add_argument("--start-idx", type=int, default=0, help="Start processing from this sample index")
    parser.add_argument("--end-idx", type=int, default=1000000, help="Process until this sample index")
    parser.add_argument("--version", default=VERSION, help="Version string for output files")
    
    args = parser.parse_args()
    
    try:
        main(
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            version=args.version
        )
    except Exception as e:
        print("Uncaught exception:", e, file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
