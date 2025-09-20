"""Upload embeddings to HuggingFace with rate limit handling."""
import os
import time
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

# Config
HF_REPO_ID = "yulia-volkova/llama-3b-outlines-embeddings_new"
RESULTS_DIR = Path(__file__).parent / "results"
BATCH_SIZE = 40  # Number of files to upload before sleeping
SLEEP_TIME = 150  # Sleep for 5 minutes between batches
MAX_RETRIES = 3  # Maximum number of retries per file
RETRY_DELAY = 3600  # Wait 1 hour after rate limit error

def get_uploaded_files():
    """Get list of files already on HuggingFace."""
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset")
        return set(f for f in files if f.endswith('.pt'))
    except Exception as e:
        print(f"Error listing repo files: {e}")
        return set()

def upload_with_retry(api, local_file, chunk_id):
    """Upload a single file with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            api.upload_file(
                path_or_fileobj=str(local_file),
                path_in_repo=f"outlines_{chunk_id:03d}.pt",
                repo_id=HF_REPO_ID,
                repo_type="dataset"
            )
            print(f"Successfully uploaded {local_file.name}")
            return True
        except HfHubHTTPError as e:
            if "429" in str(e):  # Rate limit error
                wait_time = RETRY_DELAY if attempt < MAX_RETRIES - 1 else 0
                print(f"\nRate limit hit. Waiting {wait_time//60} minutes before retry...")
                if wait_time > 0:
                    time.sleep(wait_time)
            else:
                print(f"Upload error for {local_file.name}: {e}")
                return False
    return False

def main():
    """Upload all generated embeddings to HuggingFace with rate limit handling."""
    local_dir = RESULTS_DIR / "llama-3b-outlines-embeddings_new"
    if not local_dir.exists():
        print(f"Error: Directory not found: {local_dir}")
        return

    # Initialize HF API
    api = HfApi()
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)
    
    # Get local and remote files
    local_files = sorted(local_dir.glob("outlines_*.pt"))
    print(f"Found {len(local_files)} local files")
    
    uploaded_files = get_uploaded_files()
    print(f"Found {len(uploaded_files)} files on HuggingFace")
    
    # Filter files that need uploading
    files_to_upload = []
    for local_file in local_files:
        chunk_id = int(local_file.stem.split("_")[1])
        remote_name = f"outlines_{chunk_id:03d}.pt"
        if remote_name not in uploaded_files:
            files_to_upload.append((local_file, chunk_id))
    
    print(f"\n{len(files_to_upload)} files need to be uploaded")
    if not files_to_upload:
        print("Nothing to do!")
        return
        
    # Upload in batches
    for i, (local_file, chunk_id) in enumerate(tqdm(files_to_upload, desc="Uploading files")):
        success = upload_with_retry(api, local_file, chunk_id)
        if not success:
            print(f"\nFailed to upload {local_file.name} after {MAX_RETRIES} attempts")
            continue
            
        # Sleep between batches
        if (i + 1) % BATCH_SIZE == 0 and i < len(files_to_upload) - 1:
            print(f"\nCompleted batch. Sleeping for {SLEEP_TIME//60} minutes...")
            time.sleep(SLEEP_TIME)

if __name__ == "__main__":
    main()
