"""Test script to create and upload dummy files with the desired structure."""

import os
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_test_structure(version="2.0", repo_id=None):
    """Create and upload test files with version/data/embeddings structure."""
    
    # Create local version directory with subdirs
    version_dir = f"v{version}"
    os.makedirs(f"{version_dir}/data", exist_ok=True)
    os.makedirs(f"{version_dir}/embeddings", exist_ok=True)
    
    # Create dummy files
    for i in range(2):  # Create 2 files in each directory
        # Dummy parquet file
        df = pd.DataFrame({'test': [1, 2, 3]})
        df.to_parquet(f"{version_dir}/data/chunk_{i:03d}.parquet")
        
        # Dummy pytorch tensor
        tensor = torch.tensor([1.0, 2.0, 3.0])
        torch.save(tensor, f"{version_dir}/embeddings/chunk_{i:03d}.pt")
    
    if repo_id:
        # Upload to HuggingFace
        api = HfApi(token=os.getenv("HF_TOKEN"))
        
        # Create repo if needed
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        except Exception as e:
            print(f"Note: Repository exists or error occurred: {e}")
        
        # Upload files by type to ensure both data and embeddings are handled
        print("\nUploading data files:")
        data_dir = f"{version_dir}/data"
        for file in os.listdir(data_dir):
            if file.endswith('.parquet'):
                local_path = os.path.join(data_dir, file)
                repo_path = f"{version_dir}/data/{file}"
                print(f"Uploading {local_path} -> {repo_path}")
                try:
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=repo_path,
                        repo_id=repo_id,
                        repo_type="dataset"
                    )
                    print("✓ Success")
                except Exception as e:
                    print(f"✗ Failed: {e}")

        print("\nUploading embedding files:")
        emb_dir = f"{version_dir}/embeddings"
        for file in os.listdir(emb_dir):
            if file.endswith('.pt'):
                local_path = os.path.join(emb_dir, file)
                repo_path = f"{version_dir}/embeddings/{file}"
                print(f"Uploading {local_path} -> {repo_path}")
                try:
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=repo_path,
                        repo_id=repo_id,
                        repo_type="dataset"
                    )
                    print("✓ Success")
                except Exception as e:
                    print(f"✗ Failed: {e}")
        
        print(f"\nFiles uploaded to: https://huggingface.co/datasets/{repo_id}")
    else:
        print(f"\nFiles created locally in {version_dir}/")
        print("Directory structure:")
        for root, dirs, files in os.walk(version_dir):
            level = root.replace(version_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print(f"{subindent}{f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test HuggingFace directory structure")
    parser.add_argument("--version", default="1.0", help="Version string")
    parser.add_argument("--repo-id", help="Optional: HuggingFace repo ID to upload to")
    
    args = parser.parse_args()
    create_test_structure(version=args.version, repo_id=args.repo_id)
