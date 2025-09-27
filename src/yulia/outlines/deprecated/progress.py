"""Progress tracking for outline generation pipeline."""

import os
import json
from pathlib import Path

class ProgressTracker:
    def __init__(self, version_dir: Path):
        self.version_dir = version_dir
        self.progress_file = version_dir / "progress.json"
        self.load_progress()
    
    def load_progress(self):
        """Load progress from file or initialize new progress."""
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                "last_completed_sample": -1,
                "last_completed_chunk": -1,
                "total_samples": 0,
                "completed_samples": 0
            }
            self.save_progress()
    
    def save_progress(self):
        """Save current progress to file."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def update_progress(self, completed_samples: int, chunk_idx: int = None):
        """Update progress after processing samples."""
        self.progress["completed_samples"] = completed_samples
        self.progress["last_completed_sample"] = completed_samples - 1
        if chunk_idx is not None:
            self.progress["last_completed_chunk"] = chunk_idx
        self.save_progress()
    
    def set_total_samples(self, total: int):
        """Set total number of samples to process."""
        self.progress["total_samples"] = total
        self.save_progress()
    
    def get_next_start_index(self) -> int:
        """Get the index to start processing from."""
        return max(0, self.progress["last_completed_sample"] + 1)
    
    def get_next_chunk_index(self) -> int:
        """Get the next chunk index to use."""
        return max(0, self.progress["last_completed_chunk"] + 1)
    
    def get_completion_status(self) -> str:
        """Get a string describing current completion status."""
        total = self.progress["total_samples"]
        completed = self.progress["completed_samples"]
        if total == 0:
            return "No samples processed yet"
        percentage = (completed / total) * 100
        return f"Processed {completed}/{total} samples ({percentage:.1f}%)"
