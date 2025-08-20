"""Script to delete all files from HuggingFace dataset repository."""

import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def delete_all_files(repo_id: str):
    """Delete all files from a HuggingFace dataset repository.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
    """
    api = HfApi(token=os.getenv("HF_TOKEN"))
    
    # List all files in the repository
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as e:
        print(f"Error listing files: {e}")
        return
    
    if not files:
        print("No files found in repository.")
        return
    
    print(f"Found {len(files)} files to delete:")
    for f in files:
        print(f"  - {f}")
    
    # Confirm deletion
    confirm = input("\nDo you want to delete all these files? (y/N): ")
    if confirm.lower() != 'y':
        print("Deletion cancelled")
        return
    
    # Delete files
    print("\nDeleting files...")
    for file_path in files:
        try:
            api.delete_file(
                path_in_repo=file_path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Delete {file_path}"
            )
            print(f"✓ Deleted {file_path}")
        except Exception as e:
            print(f"✗ Error deleting {file_path}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Delete all files from HuggingFace dataset")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repository ID (e.g., 'username/dataset-name')")
    
    args = parser.parse_args()
    delete_all_files(args.repo_id)