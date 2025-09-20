"""Train a linear probe to map from residual streams to SONAR embeddings."""
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

# Set up logging
log_dir = Path(__file__).parent / "logs" / "probe_training"
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"probe_training_{timestamp}.log"

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import einops
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from datasets import load_dataset

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32  # Using float32 for better stability
RESIDUALS_PATH = "/workspace/hdd_cache/tensors/llama-3b"
HF_REPO_ID = "yulia-volkova/llama-3b-outlines-embeddings_new"  # Repository with 2k embeddings (chunks 000-001)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

@dataclass
class WelfordStats:
    """Track running mean and variance using Welford's algorithm."""
    mean: torch.Tensor
    m2: torch.Tensor
    count: int

    def __init__(self, mean: torch.Tensor = None, m2: torch.Tensor = None, count: int = 0):
        self.mean = mean if mean is not None else None
        self.m2 = m2 if m2 is not None else None
        self.count = count

    def update(self, new_data: torch.Tensor):
        """Update statistics with new batch of data."""
        if self.mean is None:
            self.mean = torch.zeros_like(new_data[0])
            self.m2 = torch.zeros_like(new_data[0])
            self.count = 0
        for x in new_data:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.m2 += delta * delta2

    @property
    def std(self):
        return torch.sqrt(self.m2 / (self.count - 1) + 1e-6)

class Normalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-6)

    def restore(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + 1e-6) + self.mean

def load_and_process_chunks(chunk_ids: List[int], residuals_path: str, hf_repo_id: str):
    """Load and process data from multiple chunks, computing normalization stats along the way."""
    print("\nLoading and checking embeddings first...")
    logger.info("Loading and checking embeddings first...")
    
    # First check all embeddings
    print("\nChecking all embedding chunks:")
    print("=" * 50)
    logger.info("Checking all embedding chunks:")
    logger.info("=" * 50)
    
    for chunk_id in tqdm(chunk_ids, desc="Checking embeddings", leave=False):
        emb_file = hf_hub_download(
            repo_id=hf_repo_id,
            filename=f"outlines_{chunk_id:03d}.pt",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN")
        )
        embeds = torch.load(emb_file, map_location='cpu')
        
        if isinstance(embeds, torch.Tensor):
            msg = f"Chunk {chunk_id:03d}: shape={embeds.shape}, dtype={embeds.dtype}"
            print(msg)
            logger.info(msg)
        else:
            msg = f"Chunk {chunk_id:03d}: type={type(embeds)}, length={len(embeds)}"
            print(msg)
            logger.info(msg)
            
        assert len(embeds) == 1000, f"Expected 1000 samples in chunk {chunk_id}, got {len(embeds)}"
    
    print("=" * 50)
    print("\nNow loading and processing all data...")
    logger.info("=" * 50)
    logger.info("Now loading and processing all data...")
    
    # Initialize stats
    res_stats = WelfordStats()
    embed_stats = WelfordStats()
    
    # Prepare data
    dataset = []
    res_reshape = "layer para dim -> para layer dim"  # Reorder dimensions for the probe
    
    for chunk_id in tqdm(chunk_ids, desc="Processing chunks"):
        print(f"\nProcessing chunk {chunk_id:03d}...")
        
        # Load residuals
        res_path = Path(residuals_path) / f"res_data_{chunk_id:03d}.pt"
        print(f"\nLoading residuals from {res_path}")
        res_data = torch.load(res_path, map_location='cpu')
        print(f"Residuals: {len(res_data)} items")
        print(f"First residual: {type(res_data[0])}")
        if isinstance(res_data[0], dict):
            print(f"Keys: {res_data[0].keys()}")
            print(f"Res shape: {res_data[0]['res'].shape}")
        
        # Load embeddings - residual chunk ID maps directly to embedding chunk ID
        # Both should contain exactly 1000 items
        emb_file = hf_hub_download(
            repo_id=hf_repo_id,
            filename=f"outlines_{chunk_id:03d}.pt",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN")
        )
        print(f"\nLoading embeddings from {emb_file}")
        embeds = torch.load(emb_file, map_location='cpu').to(dtype=DTYPE)  # Convert to float32
        print(f"Embeddings: {type(embeds)}")
        if isinstance(embeds, torch.Tensor):
            print(f"Shape: {embeds.shape}")
            assert embeds.shape[0] == 1000, f"Expected 1000 embedding samples in chunk {chunk_id}, got {embeds.shape[0]}"
        else:
            print(f"Length: {len(embeds)}")
            assert len(embeds) == 1000, f"Expected 1000 embedding samples in chunk {chunk_id}, got {len(embeds)}"
            
        # Verify residuals also have 1000 samples
        assert len(res_data) == 1000, f"Expected 1000 residual samples in chunk {chunk_id}, got {len(res_data)}"
        
        assert len(res_data) == len(embeds), f"Length mismatch in chunk {chunk_id}: residuals={len(res_data)}, embeddings={len(embeds)}"
        
        # Process samples and update stats for norms
        for i, (res, embed) in enumerate(zip(res_data, embeds)):
            _id = i + 1000 * chunk_id  
            # Get residuals and take only first paragraph
            res_data = res['res'].to(dtype=DTYPE)  # Convert to float32
            first_para = res_data[:, :1, :]  # Keep only first paragraph
            res_tensor = einops.rearrange(first_para, res_reshape)  # Shape: [1, n_layers, d_model]
            
            # Print dtype info only for the first sample
            if chunk_id == chunk_ids[0] and i == 0:
                print(f"\nData types:")
                print(f"- Residuals: {res_tensor.dtype}")
                print(f"- Embeddings: {embed.dtype}")
            
            # Print shapes only for the first sample
            if chunk_id == chunk_ids[0] and i == 0:
                print(f"\nResidual shapes:")
                print(f"- Original: {res_data.shape}")
                print(f"- First paragraph: {first_para.shape}")
                print(f"- Rearranged: {res_tensor.shape}")
            
            # Update stats
            res_stats.update(res_tensor)
            embed_stats.update(embed)
            
            # Store raw tensors
            dataset.append({
                "id": _id,
                "res_data": res_tensor,  
                "embeds": embed
            })
        
        # Print info for first chunk only
        if chunk_id == chunk_ids[0]:
            print(f"\nSample shapes:")
            print(f"- Residuals: {dataset[0]['res_data'].shape} (1, n_layers, d_model)")
            print(f"- Embeddings: {dataset[0]['embeds'].shape} (d_sonar)")
    
    print(f"\nLoaded {len(dataset)} total samples")
    print(f"ID range: {dataset[0]['id']} to {dataset[-1]['id']}")
    
    # Create normalizers
    print("\nCreating normalizers...")
    res_normalizer = Normalizer(res_stats.mean, res_stats.std)
    embed_normalizer = Normalizer(embed_stats.mean, embed_stats.std)
    
    # Normalize all data
    print("Normalizing data...")
    for item in dataset:
        item['res_data'] = res_normalizer.normalize(item['res_data'])
        item['embeds'] = embed_normalizer.normalize(item['embeds'])
    
    # Verify normalization
    print("\nVerifying normalization:")
    print(f"- Residuals mean: {dataset[0]['res_data'].mean():.4f}, std: {dataset[0]['res_data'].std():.4f}")
    print(f"- Embeddings mean: {dataset[0]['embeds'].mean():.4f}, std: {dataset[0]['embeds'].std():.4f}")
    
    return dataset, res_normalizer, embed_normalizer

class OutlineDataset(Dataset):
    """Dataset for training the probe."""
    def __init__(self, residuals_path: str, hf_repo_id: str, chunk_ids: List[int], 
                 split: str = 'train', train_size: int = 800,
                 data: Optional[List[Dict]] = None,
                 res_normalizer: Optional[Normalizer] = None,
                 embed_normalizer: Optional[Normalizer] = None):
        """
        Args:
            chunk_ids: List of chunk IDs to load (e.g., [0,1,2] for chunks 000-002)
            split: Either 'train' or 'val'
            train_size: Number of samples to use for training (rest will be validation)
            data: Optional pre-loaded and normalized data
            res_normalizer: Optional pre-computed residual normalizer
            embed_normalizer: Optional pre-computed embedding normalizer
        """
        # Store normalizers
        self.res_normalizer = res_normalizer
        self.embed_normalizer = embed_normalizer
        
        # Load or use provided data
        if data is None:
            data, self.res_normalizer, self.embed_normalizer = load_and_process_chunks(
                chunk_ids, residuals_path, hf_repo_id
            )
        
        # Split data
        split_point = train_size if split == 'train' else -train_size
        self.data = data[:split_point] if split == 'train' else data[-train_size:]
        
        print(f"\n{split.title()} set summary:")
        print(f"Chunks: {[f'{i:03d}' for i in chunk_ids]}")
        print(f"Samples: {len(self.data):,}")
        print(f"ID range: {self.data[0]['id']} to {self.data[-1]['id']}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['res_data'], item['embeds'], item['id']

class LinearProbe(nn.Module):
    """Linear probe to map from residual stream to SONAR embedding."""
    def __init__(self, n_layers: int = 57, d_model: int = 3072, d_sonar: int = 1024):
        super().__init__()
        self.n_layers = n_layers
        d_in = d_model * n_layers  # Use all layers
        self.linear = nn.Linear(d_in, d_sonar)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map from residual activations to SONAR embedding.
        
        Input shape: [batch_size, n_layers, d_model]
        - batch_size is 1 (one paragraph)
        - n_layers is 57 (all transformer layers)
        - d_model is 3072 (transformer hidden size)
        
        Output shape: [batch_size, d_sonar=1024]
        - Each sample becomes a SONAR embedding
        - The linear layer maps from (n_layers * d_model) to d_sonar
        """
        # Flatten layers and model dimensions into one feature vector
        x = x.reshape(x.shape[0], -1)  # Shape: [batch, (n_layers * d_model)]
        return self.linear(x)  # Shape: [batch, d_sonar]

class ProbeTrainer:
    def __init__(
        self,
        probe: nn.Module,
        lr: float = 1e-5,
        weight_decay: float = 1e-6,
        lr_decay: float = 0.8,
        batch_size: int = 32,
        checkpoint_dir: str = "checkpoints"
    ):
        self.probe = probe
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir

        # Training components
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.probe.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1,
            gamma=self.lr_decay
        )

        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch: int, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.probe.train()
        epoch_loss = 0
        n_batches = 0
        
        logger.info(f"Starting epoch {epoch+1}")
        logger.info(f"Learning rate: {self.scheduler.get_last_lr()[0]:.2e}")

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        for batch_x, batch_y, batch_idx in pbar:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            # Forward pass
            self.optimizer.zero_grad()
            pred = self.probe(batch_x)
            loss = self.criterion(pred, batch_y)

            # Log batch statistics
            with torch.no_grad():
                mse = ((pred - batch_y) ** 2).mean().item()
                pred_mean = pred.mean().item()
                pred_std = pred.std().item()
                logger.info(f"Batch {n_batches}: MSE={mse:.4f}, Pred mean={pred_mean:.4f}, std={pred_std:.4f}")

            # Backward pass
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "Loss": f"{epoch_loss/n_batches:.4f}",
                "LR": f"{self.scheduler.get_last_lr()[0]:.2e}",
                "Idx": f"{batch_idx[0].item()}-{batch_idx[-1].item()}"
            })

        return epoch_loss / n_batches

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.probe.eval()
        val_loss = 0
        n_batches = 0
        total_mse = 0
        total_samples = 0
        all_pred_means = []
        all_pred_stds = []

        logger.info("\nStarting validation...")
        for batch_x, batch_y, batch_idx in tqdm(val_loader, desc="Validation"):
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            pred = self.probe(batch_x)
            loss = self.criterion(pred, batch_y)
            
            # Compute statistics
            mse = ((pred - batch_y) ** 2).mean().item()
            pred_mean = pred.mean().item()
            pred_std = pred.std().item()
            
            # Update running stats
            val_loss += loss.item()
            total_mse += mse * len(batch_x)
            total_samples += len(batch_x)
            all_pred_means.append(pred_mean)
            all_pred_stds.append(pred_std)
            n_batches += 1
            
            # Log batch stats
            logger.info(
                f"Val batch {n_batches}: MSE={mse:.4f}, "
                f"Pred mean={pred_mean:.4f}, std={pred_std:.4f}, "
                f"IDs: {batch_idx[0].item()}-{batch_idx[-1].item()}"
            )

        # Compute final stats
        avg_loss = val_loss / n_batches
        avg_mse = total_mse / total_samples
        avg_pred_mean = sum(all_pred_means) / len(all_pred_means)
        avg_pred_std = sum(all_pred_stds) / len(all_pred_stds)
        
        # Log validation summary
        val_summary = (
            f"\nValidation Summary:\n"
            f"- Average Loss: {avg_loss:.4f}\n"
            f"- Average MSE: {avg_mse:.4f}\n"
            f"- Average Prediction Mean: {avg_pred_mean:.4f}\n"
            f"- Average Prediction Std: {avg_pred_std:.4f}\n"
            f"- Total Batches: {n_batches}\n"
            f"- Total Samples: {total_samples}"
        )
        print(val_summary)
        logger.info(val_summary)

        return avg_loss

    def train(
        self,
        num_epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_checkpoints: bool = True
    ) -> Dict[str, List[float]]:
        """Main training loop."""
        train_losses = []
        val_losses = []

        # Log training setup
        setup_info = (
            f"\nTraining Configuration:\n"
            f"- Epochs: {num_epochs}\n"
            f"- Training batches: {len(train_loader)}\n"
            f"- Validation batches: {len(val_loader)}\n"
            f"- Initial LR: {self.lr}\n"
            f"- LR Decay: {self.lr_decay}\n"
            f"- Batch size: {self.batch_size}\n"
            f"- Weight decay: {self.weight_decay}\n"
            f"- Device: {DEVICE}\n"
            f"- Checkpoints: {self.checkpoint_dir if save_checkpoints else 'disabled'}"
        )
        print(setup_info)
        logger.info(setup_info)

        try:
            for epoch in range(num_epochs):
                # Training
                train_loss = self.train_epoch(epoch, train_loader)
                train_losses.append(train_loss)

                # Validation
                val_loss = self.validate(val_loader)
                val_losses.append(val_loss)

                # Log epoch results
                epoch_info = (
                    f"\nEpoch {epoch+1} Summary:\n"
                    f"- Train Loss: {train_loss:.4f}\n"
                    f"- Val Loss: {val_loss:.4f}\n"
                    f"- Learning Rate: {self.scheduler.get_last_lr()[0]:.2e}"
                )
                print(epoch_info)
                logger.info(epoch_info)

                # Save checkpoint
                if save_checkpoints:
                    checkpoint_path = os.path.join(
                        self.checkpoint_dir,
                        f"probe_epoch_{epoch+1}.pt"
                    )
                    self.save_checkpoint(checkpoint_path, epoch, train_loss, val_loss)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                # Step the learning rate scheduler
                self.scheduler.step()

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")

        return {
            "train_losses": train_losses,
            "val_losses": val_losses
        }

    def save_checkpoint(self, checkpoint_path: str, epoch: int, train_loss: float, val_loss: float):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.probe.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, checkpoint_path)

def train_probe(
    residuals_path: str = RESIDUALS_PATH,
    hf_repo_id: str = HF_REPO_ID,
    n_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    start_chunk: int = 0,  # Start from this chunk (inclusive)
    end_chunk: int = 0,  # End at this chunk (inclusive)
    train_size: int = 800,  # Use 800 samples for training
    checkpoint_dir: str = str(Path(RESULTS_DIR) / "probes")
):
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading datasets...")
    chunk_ids = list(range(start_chunk, end_chunk + 1))
    print(f"Using chunks: {[f'{i:03d}' for i in chunk_ids]}")
    
    # Load and process all data once
    data, res_normalizer, embed_normalizer = load_and_process_chunks(
        chunk_ids, residuals_path, hf_repo_id
    )
    
    # Create train and val datasets using the same data
    train_dataset = OutlineDataset(
        residuals_path, hf_repo_id, chunk_ids,
        split='train', train_size=train_size,
        data=data,
        res_normalizer=res_normalizer,
        embed_normalizer=embed_normalizer
    )
    
    val_dataset = OutlineDataset(
        residuals_path, hf_repo_id, chunk_ids,
        split='val', train_size=train_size,
        data=data,
        res_normalizer=res_normalizer,
        embed_normalizer=embed_normalizer
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Initialize probe model
    sample_res = train_dataset.data[0]['res_data']  # Get first residual to check shape
    n_layers, d_model = sample_res.shape[1:]  # Shape is [1, n_layers, d_model]
    print(f"\nInitializing probe:")
    print(f"- Input shape: [batch_size, {n_layers} layers, {d_model} dimensions]")
    print(f"- Output shape: [batch_size, 1024]")
    linear_probe = LinearProbe(n_layers=n_layers, d_model=d_model).to(DEVICE)
    
    trainer = ProbeTrainer(
        probe=linear_probe,
        lr=learning_rate,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir
    )
    
    losses = trainer.train(
        num_epochs=n_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        save_checkpoints=True
    )
    
    return trainer, losses

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a linear probe to map from residual streams to SONAR embeddings")
    parser.add_argument("--residuals-path", default=RESIDUALS_PATH, help="Path to residual streams")
    parser.add_argument("--hf-repo-id", default=HF_REPO_ID, help="HuggingFace repo ID for embeddings")
    parser.add_argument("--start-chunk", type=int, default=0, help="Start from this chunk (inclusive)")
    parser.add_argument("--end-chunk", type=int, default=0, help="End at this chunk (inclusive)")
    parser.add_argument("--train-size", type=int, default=800, help="Number of samples to use for training")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    args = parser.parse_args()
    
    if args.cpu:
        DEVICE = torch.device("cpu")
    
    train_probe(
        residuals_path=args.residuals_path,
        hf_repo_id=args.hf_repo_id,
        start_chunk=args.start_chunk,
        end_chunk=args.end_chunk,
        train_size=args.train_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs
    )