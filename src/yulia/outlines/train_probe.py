"""Core: Train a linear probe to map from residual streams to SONAR embeddings.

- Normalization stats (mean/std) are computed on a subset of chunks (default: first 10).
- Training/validation use a streaming dataset with a tiny LRU chunk cache (no full materialization).
- Global fixed split: default 90% train, 10% val across ALL selected samples.
- Pluggable Tracker protocol (NoOp by default). Use a WandbTracker adapter to log to W&B.
"""
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple
from dataclasses import dataclass

# -------- logging setup --------
log_dir = Path(__file__).parent / "logs" / "probe_training"
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"probe_training_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# -------- imports --------
import torch
import torch.nn as nn
import einops
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# -------- config defaults --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
RESIDUALS_PATH = "/workspace/hdd_cache/tensors/llama-3b"
HF_REPO_ID = "yulia-volkova/llama-3b-outlines-embeddings_new"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

SAMPLES_PER_CHUNK = 1000  # invariant used throughout

# ================= Tracker protocol (pluggable logger) ======================
class Tracker(Protocol):
    def on_start(self, run_config: Dict): ...
    def define_metrics(self): ...
    def watch(self, model: nn.Module): ...
    def log(self, data: Dict, step: Optional[int] = None): ...
    def log_artifact(self, path: str, name: str, type_: str, description: str = ""): ...
    def finish(self): ...

class NoOpTracker:
    def on_start(self, run_config: Dict): pass
    def define_metrics(self): pass
    def watch(self, model: nn.Module): pass
    def log(self, data: Dict, step: Optional[int] = None): pass
    def log_artifact(self, path: str, name: str, type_: str, description: str = ""): pass
    def finish(self): pass
# ===========================================================================

@dataclass
class WelfordStats:
    mean: torch.Tensor
    m2: torch.Tensor
    count: int
    def __init__(self, mean: torch.Tensor = None, m2: torch.Tensor = None, count: int = 0):
        self.mean = mean if mean is not None else None
        self.m2 = m2 if m2 is not None else None
        self.count = count
    def update(self, new_data: torch.Tensor):
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
        # add epsilon for stability; guard count
        denom = max(self.count - 1, 1)
        return torch.sqrt(self.m2 / denom + 1e-6)

class Normalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + 1e-6)
    def restore(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + 1e-6) + self.mean

# ------------------ Pass 1: compute stats on a subset of chunks --------------------
def compute_normalizers(
    norm_chunk_ids: List[int],
    residuals_path: str,
    hf_repo_id: str,
) -> Tuple[Normalizer, Normalizer]:
    """Compute mean/std using Welford over a small subset of chunks."""
    print("\nComputing normalization stats (subset)...")
    logger.info("Computing normalization stats (subset)...")

    res_stats = WelfordStats()
    embed_stats = WelfordStats()
    res_reshape = "layer para dim -> para layer dim"

    for chunk_id in tqdm(norm_chunk_ids, desc="Stats chunks"):
        # residuals
        res_path = Path(residuals_path) / f"res_data_{chunk_id:03d}.pt"
        res_data_list = torch.load(res_path, map_location="cpu")
        assert len(res_data_list) == SAMPLES_PER_CHUNK, f"Residuals count mismatch in chunk {chunk_id}"

        # embeddings
        emb_file = hf_hub_download(
            repo_id=hf_repo_id,
            filename=f"outlines_{chunk_id:03d}.pt",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        embeds = torch.load(emb_file, map_location="cpu").to(dtype=DTYPE)
        n_embeds = embeds.shape[0] if isinstance(embeds, torch.Tensor) else len(embeds)
        assert n_embeds == SAMPLES_PER_CHUNK, f"Embeds count mismatch in chunk {chunk_id}"

        for res, embed in zip(res_data_list, embeds):
            res_all = res["res"].to(dtype=DTYPE)            # [n_layers, n_para, d_model]
            first_para = res_all[:, :1, :]                  # keep first paragraph
            res_tensor = einops.rearrange(first_para, res_reshape)  # [1, n_layers, d_model]
            res_stats.update(res_tensor)
            embed_stats.update(embed)

    res_normalizer = Normalizer(res_stats.mean, res_stats.std)
    embed_normalizer = Normalizer(embed_stats.mean, embed_stats.std)
    return res_normalizer, embed_normalizer

# ------------------ Pass 2: streaming dataset (global 90/10 split) ------------------
class StreamingOutlineDataset(Dataset):
    """Stream samples chunk-by-chunk (LRU cache), normalize on-the-fly, global split."""

    def __init__(
        self,
        residuals_path: str,
        hf_repo_id: str,
        chunk_ids: List[int],
        split: str = "train",
        train_frac: float = 0.9,
        res_normalizer: Optional[Normalizer] = None,
        embed_normalizer: Optional[Normalizer] = None,
        max_cached_chunks: int = 2,
    ):
        assert split in {"train", "val"}
        assert 0.0 < train_frac < 1.0

        self.residuals_path = Path(residuals_path)
        self.hf_repo_id = hf_repo_id
        self.chunk_ids = list(chunk_ids)
        self.res_norm = res_normalizer
        self.emb_norm = embed_normalizer
        self.max_cached = max_cached_chunks

        # Build global index space (1000 samples per chunk)
        self.total = len(self.chunk_ids) * SAMPLES_PER_CHUNK
        split_point = int(train_frac * self.total)

        if split == "train":
            idxs = list(range(0, split_point))
        else:
            idxs = list(range(split_point, self.total))
        self.indices = idxs

        # very small LRU cache: {chunk_id: (res_list, embeds_tensor)}
        self._cache: Dict[int, Tuple[List[Dict], torch.Tensor]] = {}
        self._cache_order: List[int] = []

        # for rearrange
        self._reshape = "layer para dim -> para layer dim"

        print(f"\n{split.title()} set summary:")
        print(f"Chunks: {[f'{i:03d}' for i in self.chunk_ids]}")
        print(f"Samples: {len(self.indices):,} / Total: {self.total:,}")
        if self.indices:
            print(f"Global idx range: {self.indices[0]}..{self.indices[-1]}")

    def __len__(self) -> int:
        return len(self.indices)

    def _idx_to_chunk_pos(self, global_idx: int) -> Tuple[int, int]:
        chunk_offset = global_idx // SAMPLES_PER_CHUNK
        sample_in_chunk = global_idx % SAMPLES_PER_CHUNK
        chunk_id = self.chunk_ids[chunk_offset]
        return chunk_id, sample_in_chunk

    def _ensure_in_cache(self, chunk_id: int):
        if chunk_id in self._cache:
            # bump in LRU
            self._cache_order.remove(chunk_id)
            self._cache_order.append(chunk_id)
            return

        # load residuals
        res_path = self.residuals_path / f"res_data_{chunk_id:03d}.pt"
        res_list = torch.load(res_path, map_location="cpu")
        assert len(res_list) == SAMPLES_PER_CHUNK

        # load embeddings
        emb_file = hf_hub_download(
            repo_id=self.hf_repo_id,
            filename=f"outlines_{chunk_id:03d}.pt",
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        embeds = torch.load(emb_file, map_location="cpu").to(dtype=DTYPE)
        n_embeds = embeds.shape[0] if isinstance(embeds, torch.Tensor) else len(embeds)
        assert n_embeds == SAMPLES_PER_CHUNK

        # insert into cache
        self._cache[chunk_id] = (res_list, embeds)
        self._cache_order.append(chunk_id)

        # evict if needed
        if len(self._cache_order) > self.max_cached:
            evict_id = self._cache_order.pop(0)
            self._cache.pop(evict_id, None)

    def __getitem__(self, i: int):
        global_idx = self.indices[i]
        chunk_id, pos = self._idx_to_chunk_pos(global_idx)
        self._ensure_in_cache(chunk_id)
        res_list, embeds = self._cache[chunk_id]

        # fetch item
        res_all = res_list[pos]["res"].to(dtype=DTYPE)  # [n_layers, n_para, d_model]
        first_para = res_all[:, :1, :]
        res_tensor = einops.rearrange(first_para, self._reshape)  # [1, n_layers, d_model]
        emb = embeds[pos]

        # normalize on the fly
        if self.res_norm is not None:
            res_tensor = self.res_norm.normalize(res_tensor)
        if self.emb_norm is not None:
            emb = self.emb_norm.normalize(emb)

        return res_tensor, emb, torch.tensor(global_idx, dtype=torch.long)

# -------------------------- Model & Trainer -------------------------------
class LinearProbe(nn.Module):
    def __init__(self, n_layers: int = 57, d_model: int = 3072, d_sonar: int = 1024):
        super().__init__()
        self.n_layers = n_layers
        self.linear = nn.Linear(d_model * n_layers, d_sonar)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        return self.linear(x)

class ProbeTrainer:
    def __init__(self, probe: nn.Module, lr: float = 1e-5, weight_decay: float = 1e-6,
                 lr_decay: float = 0.8, batch_size: int = 32, checkpoint_dir: str = "checkpoints",
                 tracker: Optional[Tracker] = None):
        self.probe = probe
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.lr_decay)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.tracker = tracker or NoOpTracker()
        self.global_step = 0

    def train_epoch(self, epoch: int, train_loader: DataLoader) -> float:
        self.probe.train()
        epoch_loss, n_batches = 0.0, 0
        logger.info(f"Starting epoch {epoch+1} | LR {self.scheduler.get_last_lr()[0]:.2e}")
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        for batch_x, batch_y, batch_idx in pbar:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            self.optimizer.zero_grad()
            pred = self.probe(batch_x)
            loss = self.criterion(pred, batch_y)
            with torch.no_grad():
                mse = ((pred - batch_y)**2).mean().item()
                pred_mean = pred.mean().item()
                pred_std = pred.std().item()
                logger.info(f"Batch {n_batches}: MSE={mse:.4f}, Pred mean={pred_mean:.4f}, std={pred_std:.4f}")
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item(); n_batches += 1; self.global_step += 1

            # tracker: per-step
            self.tracker.log({
                "global_step": self.global_step,
                "train/loss": loss.item(),
                "train/mse": mse,
                "train/pred_mean": pred_mean,
                "train/pred_std": pred_std,
                "train/lr": self.scheduler.get_last_lr()[0],
                "train/epoch": epoch + 1,
                "train/idx_start": batch_idx[0].item(),
                "train/idx_end": batch_idx[-1].item(),
            }, step=self.global_step)

            pbar.set_postfix({"Loss": f"{epoch_loss/n_batches:.4f}",
                              "LR": f"{self.scheduler.get_last_lr()[0]:.2e}",
                              "Idx": f"{batch_idx[0].item()}-{batch_idx[-1].item()}"})
        return epoch_loss / n_batches

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        self.probe.eval()
        val_loss = 0.0; n_batches = 0
        total_mse = 0.0; total_samples = 0
        all_pred_means, all_pred_stds = [], []
        logger.info("\nStarting validation...")
        for batch_x, batch_y, batch_idx in tqdm(val_loader, desc="Validation"):
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            pred = self.probe(batch_x)
            loss = self.criterion(pred, batch_y)
            mse = ((pred - batch_y)**2).mean().item()
            pred_mean = pred.mean().item(); pred_std = pred.std().item()
            val_loss += loss.item(); total_mse += mse * len(batch_x); total_samples += len(batch_x)
            all_pred_means.append(pred_mean); all_pred_stds.append(pred_std); n_batches += 1
            logger.info(f"Val batch {n_batches}: MSE={mse:.4f}, Pred mean={pred_mean:.4f}, std={pred_std:.4f}, IDs: {batch_idx[0].item()}-{batch_idx[-1].item()}")
        avg_loss = val_loss / n_batches
        avg_mse = total_mse / total_samples
        avg_pred_mean = sum(all_pred_means) / len(all_pred_means)
        avg_pred_std = sum(all_pred_stds) / len(all_pred_stds)
        # tracker: val epoch summary
        self.tracker.log({
            "val/loss": avg_loss,
            "val/mse": avg_mse,
            "val/pred_mean": avg_pred_mean,
            "val/pred_std": avg_pred_std,
        })
        summary = (f"\nValidation Summary:\n- Average Loss: {avg_loss:.4f}\n- Average MSE: {avg_mse:.4f}"
                   f"\n- Average Prediction Mean: {avg_pred_mean:.4f}\n- Average Prediction Std: {avg_pred_std:.4f}"
                   f"\n- Total Batches: {n_batches}\n- Total Samples: {total_samples}")
        print(summary); logger.info(summary)
        return avg_loss

    def train(self, num_epochs: int, train_loader: DataLoader, val_loader: DataLoader,
              save_checkpoints: bool = True) -> Dict[str, List[float]]:
        train_losses, val_losses = [], []
        setup = (f"\nTraining Configuration:\n- Epochs: {num_epochs}\n- Training batches: {len(train_loader)}"
                 f"\n- Validation batches: {len(val_loader)}\n- Initial LR: {self.lr}\n- LR Decay: {self.lr_decay}"
                 f"\n- Batch size: {self.batch_size}\n- Weight decay: {self.weight_decay}\n- Device: {DEVICE}"
                 f"\n- Checkpoints: {self.checkpoint_dir if save_checkpoints else 'disabled'}")
        print(setup); logger.info(setup)
        try:
            for epoch in range(num_epochs):
                tr = self.train_epoch(epoch, train_loader); train_losses.append(tr)
                vl = self.validate(val_loader); val_losses.append(vl)
                epoch_info = (f"\nEpoch {epoch+1} Summary:\n- Train Loss: {tr:.4f}\n- Val Loss: {vl:.4f}"
                              f"\n- Learning Rate: {self.scheduler.get_last_lr()[0]:.2e}")
                print(epoch_info); logger.info(epoch_info)
                self.tracker.log({"epoch": epoch + 1, "epoch/train_loss": tr, "epoch/val_loss": vl,
                                  "epoch/lr": self.scheduler.get_last_lr()[0]})
                if save_checkpoints:
                    ckpt = os.path.join(self.checkpoint_dir, f"probe_epoch_{epoch+1}.pt")
                    self.save_checkpoint(ckpt, epoch, tr, vl)
                    logger.info(f"Saved checkpoint to {ckpt}")
                    self.tracker.log_artifact(ckpt, name=f"checkpoint-epoch-{epoch+1}",
                                              type_="model", description=f"Linear probe after epoch {epoch+1}")
                self.scheduler.step()
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        return {"train_losses": train_losses, "val_losses": val_losses}

    def save_checkpoint(self, checkpoint_path: str, epoch: int, train_loss: float, val_loss: float):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.probe.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, checkpoint_path)

# ---------------------- helpers & entrypoint ------------------------------
def _save_normalizers(res_normalizer: Normalizer, embed_normalizer: Normalizer, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "res_normalizer.pt"
    emb_path = out_dir / "embed_normalizer.pt"
    torch.save({"mean": res_normalizer.mean, "std": res_normalizer.std}, res_path)
    torch.save({"mean": embed_normalizer.mean, "std": embed_normalizer.std}, emb_path)
    return str(res_path), str(emb_path)

def train_probe(residuals_path: str = RESIDUALS_PATH, hf_repo_id: str = HF_REPO_ID,
                n_epochs: int = 10, batch_size: int = 32, learning_rate: float = 1e-4,
                start_chunk: int = 0, end_chunk: int = 0,
                checkpoint_dir: str = str(Path(RESULTS_DIR) / "probes"),
                tracker: Optional[Tracker] = None,
                norm_chunks: int = 10,
                train_frac: float = 0.9,
                max_cached_chunks: int = 2):
    """Main training function (streaming + global split)."""
    tracker = tracker or NoOpTracker()
    checkpoint_dir = Path(checkpoint_dir); checkpoint_dir.mkdir(parents=True, exist_ok=True)

    chunk_ids = list(range(start_chunk, end_chunk + 1))
    assert len(chunk_ids) > 0, "No chunks selected."
    print("Selected chunks:", [f"{i:03d}" for i in chunk_ids])

    # Pass 1: compute normalizers on a small subset
    norm_chunk_ids = chunk_ids[:min(norm_chunks, len(chunk_ids))]
    res_norm, emb_norm = compute_normalizers(norm_chunk_ids, residuals_path, hf_repo_id)

    # Save & (optionally) log normalizers
    res_p, emb_p = _save_normalizers(res_norm, emb_norm, Path(RESULTS_DIR) / "normalizers")
    tracker.log_artifact(res_p, name="res_normalizer.pt", type_="asset", description="Residual normalizer")
    tracker.log_artifact(emb_p, name="embed_normalizer.pt", type_="asset", description="Embedding normalizer")

    # Pass 2: streaming datasets (no full materialization). Global fixed split 90/10 by default.
    train_dataset = StreamingOutlineDataset(
        residuals_path=residuals_path,
        hf_repo_id=hf_repo_id,
        chunk_ids=chunk_ids,
        split="train",
        train_frac=train_frac,
        res_normalizer=res_norm,
        embed_normalizer=emb_norm,
        max_cached_chunks=max_cached_chunks,
    )
    val_dataset = StreamingOutlineDataset(
        residuals_path=residuals_path,
        hf_repo_id=hf_repo_id,
        chunk_ids=chunk_ids,
        split="val",
        train_frac=train_frac,
        res_normalizer=res_norm,
        embed_normalizer=emb_norm,
        max_cached_chunks=max_cached_chunks,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # model init: infer shapes from a sample
    sample_res, _, _ = train_dataset[0]
    n_layers, d_model = sample_res.shape[1], sample_res.shape[2]
    print(f"\nInitializing probe:\n- Input: [batch, {n_layers} layers, {d_model} dims]\n- Output: 1024")
    probe = LinearProbe(n_layers=n_layers, d_model=d_model).to(DEVICE)

    run_config = {
        "n_epochs": n_epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "start_chunk": start_chunk, "end_chunk": end_chunk,
        "device": str(DEVICE), "dtype": str(DTYPE), "hf_repo_id": hf_repo_id,
        "residuals_path": residuals_path, "n_layers": int(n_layers), "d_model": int(d_model),
        "d_sonar": 1024, "num_train_batches": len(train_loader), "num_val_batches": len(val_loader),
        "norm_chunks": norm_chunks, "norm_chunk_ids": norm_chunk_ids,
        "train_frac": train_frac, "max_cached_chunks": max_cached_chunks,
        "total_samples": len(chunk_ids) * SAMPLES_PER_CHUNK,
    }
    tracker.on_start(run_config); tracker.define_metrics(); tracker.watch(probe)

    trainer = ProbeTrainer(probe=probe, lr=learning_rate, batch_size=batch_size,
                           checkpoint_dir=str(checkpoint_dir), tracker=tracker)
    losses = trainer.train(num_epochs=n_epochs, train_loader=train_loader, val_loader=val_loader, save_checkpoints=True)
    tracker.finish()
    return trainer, losses

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Core probe trainer (streaming; global 90/10 split).")
    parser.add_argument("--residuals-path", default=RESIDUALS_PATH)
    parser.add_argument("--hf-repo-id", default=HF_REPO_ID)
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--end-chunk", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--train-frac", type=float, default=0.9,
                        help="Global train fraction (0..1). Default 0.9 → 90% train / 10% val.")
    parser.add_argument("--norm-chunks", type=int, default=10,
                        help="Number of initial chunks to use for normalization stats (default: 10).")
    parser.add_argument("--max-cached-chunks", type=int, default=2,
                        help="LRU cache size in chunks (default: 2).")
    parser.add_argument("--checkpoint-dir", type=str, default=str(Path(RESULTS_DIR) / "probes"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.cpu:
        DEVICE = torch.device("cpu")

    train_probe(residuals_path=args.residuals_path, hf_repo_id=args.hf_repo_id,
                start_chunk=args.start_chunk, end_chunk=args.end_chunk,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                n_epochs=args.n_epochs, train_frac=args.train_frac,
                norm_chunks=args.norm_chunks, max_cached_chunks=args.max_cached_chunks,
                checkpoint_dir=args.checkpoint_dir)
