"""
Train a linear probe to map from residual streams to SONAR embeddings.

Key features
- Streaming from HF *or* local fallback (per-chunk). Residuals from HF repo
  `nickypro/fineweb-llama3b-residuals` or local dir, embeddings from HF repo
  `yulia-volkova/llama-3b-outlines-embeddings_new` or local dir.
- Any chunk with residuals OR embeddings length != 1000 is skipped entirely (norm + train/val).
- Normalization (Welford) computed once on a small subset (`--norm-chunks`).
- Epoch-wise chunk selection:
    * If --chunks-per-epoch is unset → use ALL chunks every epoch, in ascending order (no rotation).
    * If set → use contiguous, ascending slices; cycle across epochs in order (no shuffle).
- Tiny LRU cache: keep at most `--max-cached-chunks` chunks in RAM.
- Intermittent tests: fast "audit" eval during training on a fixed, tiny val set.
- Pluggable Tracker protocol; works with your WandbTracker via wandb_tracker.py.
- Checkpoint resume: `--resume-from` restores model/optim/scheduler and continues with the next epoch.
- HF cache auto-clean after each epoch (and optionally right after normalization).
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

HF_REPO_ID_RESIDUALS = "nickypro/fineweb-llama3b-residuals"
HF_REPO_ID_EMBEDS    = "yulia-volkova/llama-3b-outlines-embeddings_new"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SAMPLES_PER_CHUNK = 1000  # invariant

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

# ---------------------- HF OR LOCAL loaders (residuals & embeds) ---------------------
def _try_load_local(local_dir: Optional[str], filename: str):
    if not local_dir:
        return None
    p = Path(local_dir) / filename
    if p.exists():
        return torch.load(p, map_location="cpu")
    return None

def _try_load_hf(repo_id: str, filename: str, hf_token: Optional[str], force: bool = False):
    fpath = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=hf_token,
        force_download=force,
    )
    return torch.load(fpath, map_location="cpu")

def _load_with_fallback(
    fname: str,
    prefer_local: bool,
    local_dir: Optional[str],
    hf_repo_id: Optional[str],
    hf_token: Optional[str],
    cast_dtype: Optional[torch.dtype] = None,
    retry_force_hf_on_failure: bool = False,
):
    """
    Try to load torch .pt file from:
      1) local_dir (if prefer_local=True)
      2) HF repo
      3) local_dir (if prefer_local=False and HF failed)
    Optionally cast tensors to DTYPE.
    """
    err_msgs = []

    def _maybe_cast(x):
        if cast_dtype is None:
            return x
        # try to cast tensors in sensible structures
        try:
            if isinstance(x, torch.Tensor):
                return x.to(dtype=cast_dtype)
            if isinstance(x, list) and len(x) > 0 and isinstance(x[0], torch.Tensor):
                return torch.stack(x).to(dtype=cast_dtype)
        except Exception:
            pass
        return x

    # order A then B
    order = ["local", "hf"] if prefer_local else ["hf", "local"]

    for source in order:
        try:
            if source == "local":
                obj = _try_load_local(local_dir, fname)
                if obj is not None:
                    return _maybe_cast(obj), f"local:{local_dir}/{fname}"
            else:
                if hf_repo_id:
                    obj = _try_load_hf(hf_repo_id, fname, hf_token, force=False)
                    return _maybe_cast(obj), f"hf:{hf_repo_id}/{fname}"
        except Exception as e:
            err_msgs.append(f"{source} error: {e}")

    # Optional forced HF retry (to bypass a corrupt cache entry)
    if not prefer_local and hf_repo_id and retry_force_hf_on_failure:
        try:
            obj = _try_load_hf(hf_repo_id, fname, hf_token, force=True)
            return _maybe_cast(obj), f"hf-forced:{hf_repo_id}/{fname}"
        except Exception as e:
            err_msgs.append(f"hf forced error: {e}")

    raise RuntimeError(f"Failed to load {fname} from local or HF.\n" + "\n".join(err_msgs))

def _load_residuals_chunk(
    chunk_id: int,
    hf_repo_id: str,
    hf_token: Optional[str],
    local_dir: Optional[str],
    prefer_local: bool,
):
    fname = f"res_data_{chunk_id:03d}.pt"
    obj, src = _load_with_fallback(
        fname=fname,
        prefer_local=prefer_local,
        local_dir=local_dir,
        hf_repo_id=hf_repo_id,
        hf_token=hf_token,
        cast_dtype=None,
        retry_force_hf_on_failure=True,
    )
    logger.info(f"[load] residuals {chunk_id:03d} ← {src}")
    return obj

def _load_embeddings_chunk(
    chunk_id: int,
    hf_repo_id: str,
    hf_token: Optional[str],
    local_dir: Optional[str],
    prefer_local: bool,
) -> torch.Tensor:
    fname = f"outlines_{chunk_id:03d}.pt"
    obj, src = _load_with_fallback(
        fname=fname,
        prefer_local=prefer_local,
        local_dir=local_dir,
        hf_repo_id=hf_repo_id,
        hf_token=hf_token,
        cast_dtype=DTYPE,  # cast embeddings
        retry_force_hf_on_failure=True,
    )
    logger.info(f"[load] embeddings {chunk_id:03d} ← {src}")
    return obj

# ====================== Normalization (Welford) ============================
@dataclass
class WelfordStats:
    mean: torch.Tensor = None
    m2: torch.Tensor = None
    count: int = 0
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
    hf_repo_residuals: str,
    hf_repo_embeds: str,
    hf_token: Optional[str],
    local_residuals_dir: Optional[str],
    local_embeds_dir: Optional[str],
    prefer_local: bool,
) -> Tuple[Normalizer, Normalizer]:
    """
    Compute mean/std using Welford over a small subset of chunks.
    Skip any chunk where residuals or embeddings length != 1000.
    """
    print("\nComputing normalization stats (subset)...")
    logger.info("Computing normalization stats (subset)...")

    res_stats = WelfordStats()
    embed_stats = WelfordStats()
    res_reshape = "layer para dim -> para layer dim"

    used, skipped = 0, []

    for chunk_id in tqdm(norm_chunk_ids, desc="Stats chunks"):
        try:
            res_list = _load_residuals_chunk(
                chunk_id, hf_repo_residuals, hf_token, local_residuals_dir, prefer_local
            )
            embeds   = _load_embeddings_chunk(
                chunk_id, hf_repo_embeds, hf_token, local_embeds_dir, prefer_local
            )

            n_res = len(res_list)
            n_emb = embeds.shape[0] if isinstance(embeds, torch.Tensor) else len(embeds)

            if n_res != SAMPLES_PER_CHUNK or n_emb != SAMPLES_PER_CHUNK:
                logger.warning(f"[norm] skip chunk {chunk_id:03d}: res={n_res}, emb={n_emb} (expected 1000).")
                skipped.append(chunk_id); continue

            for res, embed in zip(res_list, embeds):
                res_all = res["res"].to(dtype=DTYPE)                  # [n_layers, n_para, d_model]
                first_para = res_all[:, :1, :]                        # first paragraph only
                res_tensor = einops.rearrange(first_para, res_reshape)  # [1, n_layers, d_model]
                res_stats.update(res_tensor)
                embed_stats.update(embed)

            used += 1

        except Exception as e:
            logger.warning(f"[norm] failed chunk {chunk_id:03d}: {e}. Skipping.")
            skipped.append(chunk_id); continue

    if used == 0:
        raise RuntimeError("No valid chunks for normalization (after skipping mismatches).")

    logger.info(f"[norm] used {used}/{len(norm_chunk_ids)} chunks; skipped={skipped}")

    res_normalizer = Normalizer(res_stats.mean, res_stats.std)
    embed_normalizer = Normalizer(embed_stats.mean, embed_stats.std)
    return res_normalizer, embed_normalizer

# ------------------ HF cache cleaning utilities ----------------------------
def _hf_cache_dir() -> Path:
    base = os.environ.get("HF_HOME")
    return Path(base) if base else Path.home() / ".cache" / "huggingface"

def _dir_size_gb(root: Path) -> float:
    total = 0
    try:
        for dp, _dn, fnames in os.walk(root):
            for f in fnames:
                try:
                    total += (Path(dp) / f).stat().st_size
                except Exception:
                    pass
    except Exception:
        return 0.0
    return total / (1024 ** 3)

def maybe_clean_hf_cache(threshold_gb: float, tracker: Optional[Tracker] = None) -> None:
    """
    If the HF cache directory exceeds threshold_gb, wipe it.
    Logs size to tracker as cache/size_gb and cache/cleaned.
    """
    cache = _hf_cache_dir()
    size_gb = _dir_size_gb(cache) if cache.exists() else 0.0
    logger.info(f"[cache] HF cache at {cache} ~ {size_gb:.2f} GB")
    # only log if tracker was already initialized by caller
    if tracker is not None and getattr(tracker, "_run", None):
        try:
            tracker.log({"cache/size_gb": float(size_gb)})
        except Exception:
            pass

    if size_gb <= threshold_gb:
        return

    logger.warning(f"[cache] Exceeds {threshold_gb:.1f} GB → cleaning HF cache at {cache}")

    cleaned = False
    try:
        from huggingface_hub import delete_cache_dir
        delete_cache_dir(cache_dir=str(cache))
        cleaned = True
    except Exception:
        pass

    if not cleaned:
        try:
            for child in cache.glob("*"):
                try:
                    if child.is_dir():
                        import shutil
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
                except Exception:
                    pass
            cleaned = True
        except Exception as e:
            logger.error(f"[cache] Failed to clean HF cache: {e}")

    if cleaned and tracker is not None and getattr(tracker, "_run", None):
        try:
            tracker.log({"cache/cleaned": 1})
        except Exception:
            pass

# ------------------ Streaming dataset (global 90/10 split) ------------------
class StreamingOutlineDataset(Dataset):
    """Stream samples chunk-by-chunk from HF or local (LRU cache), normalize on-the-fly, global split.
       Entire chunks are marked BAD and skipped if res/emb length != 1000.
    """

    def __init__(
        self,
        hf_repo_residuals: str,
        hf_repo_embeds: str,
        chunk_ids: List[int],
        split: str = "train",
        train_frac: float = 0.9,
        res_normalizer: Optional[Normalizer] = None,
        embed_normalizer: Optional[Normalizer] = None,
        max_cached_chunks: int = 2,
        hf_token: Optional[str] = None,
        limit_layers: Optional[int] = None,
        local_residuals_dir: Optional[str] = None,
        local_embeds_dir: Optional[str] = None,
        prefer_local: bool = False,
    ):
        assert split in {"train", "val"}
        assert 0.0 < train_frac < 1.0

        self.hf_repo_residuals = hf_repo_residuals
        self.hf_repo_embeds = hf_repo_embeds
        self.chunk_ids = list(chunk_ids)
        self.res_norm = res_normalizer
        self.emb_norm = embed_normalizer
        self.max_cached = max_cached_chunks
        self.hf_token = hf_token
        self.limit_layers = limit_layers
        self.local_residuals_dir = local_residuals_dir
        self.local_embeds_dir = local_embeds_dir
        self.prefer_local = prefer_local

        # Build global index space (1000 samples per chunk)
        self.total = len(self.chunk_ids) * SAMPLES_PER_CHUNK
        split_point = int(train_frac * self.total)

        self.indices = list(range(0, split_point)) if split == "train" else list(range(split_point, self.total))

        # very small LRU cache: {chunk_id: (res_list, embeds_tensor)}
        self._cache: Dict[int, Tuple[List[Dict], torch.Tensor]] = {}
        self._cache_order: List[int] = []
        self._bad_chunks: set[int] = set()

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
        if chunk_id in self._bad_chunks:
            return

        if chunk_id in self._cache:
            # bump LRU
            self._cache_order.remove(chunk_id)
            self._cache_order.append(chunk_id)
            return

        try:
            res_list = _load_residuals_chunk(
                chunk_id,
                self.hf_repo_residuals,
                self.hf_token,
                self.local_residuals_dir,
                self.prefer_local,
            )
            embeds   = _load_embeddings_chunk(
                chunk_id,
                self.hf_repo_embeds,
                self.hf_token,
                self.local_embeds_dir,
                self.prefer_local,
            )

            n_res = len(res_list)
            n_emb = embeds.shape[0] if isinstance(embeds, torch.Tensor) else len(embeds)

            if n_res != SAMPLES_PER_CHUNK or n_emb != SAMPLES_PER_CHUNK:
                logger.warning(f"[data] marking chunk {chunk_id:03d} BAD: res={n_res}, emb={n_emb} (expected 1000).")
                self._bad_chunks.add(chunk_id)
                return

            # valid → cache it
            self._cache[chunk_id] = (res_list, embeds)
            self._cache_order.append(chunk_id)

            # evict if needed
            if len(self._cache_order) > self.max_cached:
                evict_id = self._cache_order.pop(0)
                self._cache.pop(evict_id, None)

        except Exception as e:
            logger.warning(f"[data] failed to load chunk {chunk_id:03d}: {e}. Marking as BAD.")
            self._bad_chunks.add(chunk_id)

    def _next_good_chunk(self, start_chunk_id: int) -> Optional[int]:
        """Return the first good chunk at or after start_chunk_id, scanning circularly."""
        if not self.chunk_ids:
            return None
        try:
            start_idx = self.chunk_ids.index(start_chunk_id)
        except ValueError:
            start_idx = 0
        n = len(self.chunk_ids)
        for k in range(n):
            cid = self.chunk_ids[(start_idx + k) % n]
            if cid in self._bad_chunks:
                continue
            # ensure cache/validity
            self._ensure_in_cache(cid)
            if cid not in self._bad_chunks and cid in self._cache:
                return cid
        return None

    def __getitem__(self, i: int):
        global_idx = self.indices[i]
        chunk_id, pos = self._idx_to_chunk_pos(global_idx)

        # ensure we know status of this chunk
        self._ensure_in_cache(chunk_id)

        # if bad or not cached → pick next good chunk
        if chunk_id in self._bad_chunks or chunk_id not in self._cache:
            alt = self._next_good_chunk(chunk_id)
            if alt is None:
                raise RuntimeError("No valid chunks available (all marked bad).")
            pos = pos % SAMPLES_PER_CHUNK
            chunk_id = alt

        res_list, embeds = self._cache[chunk_id]
        # valid chunks have exactly 1000 rows
        res_all = res_list[pos]["res"].to(dtype=DTYPE)  # [n_layers, n_para, d_model]
        first_para = res_all[:, :1, :]
        res_tensor = einops.rearrange(first_para, self._reshape)  # [1, n_layers, d_model]

        # restrict to first N layers if requested
        if hasattr(self, "limit_layers") and self.limit_layers is not None:
            res_tensor = res_tensor[:, :self.limit_layers, :]

        emb = embeds[pos]

        # normalize
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

    def train_epoch(self, epoch: int, train_loader: DataLoader,
                    audit_loader: Optional[DataLoader] = None, eval_every: int = 1000) -> float:
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

            # intermittent quick eval on a tiny fixed audit set
            if audit_loader is not None and self.global_step % eval_every == 0:
                quick = self.quick_eval(audit_loader, max_batches=5)
                self.tracker.log({**quick, "train/epoch": epoch + 1, "global_step": self.global_step},
                                 step=self.global_step)

            pbar.set_postfix({"Loss": f"{epoch_loss/n_batches:.4f}",
                              "LR": f"{self.scheduler.get_last_lr()[0]:.2e}",
                              "Idx": f"{batch_idx[0].item()}-{batch_idx[-1].item()}"})
        return epoch_loss / n_batches

    @torch.no_grad()
    def quick_eval(self, loader: DataLoader, max_batches: int = 5) -> Dict[str, float]:
        """Fast sanity-check: average loss/MSE/cosine over at most `max_batches` from `loader`."""
        self.probe.eval()
        n, loss_sum, mse_sum, cos_sum = 0, 0.0, 0.0, 0.0
        cos = torch.nn.CosineSimilarity(dim=1)
        for b, (x, y, _) in enumerate(loader):
            if b >= max_batches: break
            x = x.to(DEVICE); y = y.to(DEVICE)
            pred = self.probe(x)
            loss = self.criterion(pred, y)
            mse = ((pred - y)**2).mean()
            loss_sum += loss.item()
            mse_sum += mse.item()
            cos_sum += cos(pred, y).mean().item()
            n += 1
        if n == 0:
            return {"quick/loss": float("nan"), "quick/mse": float("nan"), "quick/cos": float("nan")}
        return {"quick/loss": loss_sum/n, "quick/mse": mse_sum/n, "quick/cos": cos_sum/n}

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

# ---------------------- helpers ------------------------------
def _save_normalizers(res_normalizer: Normalizer, embed_normalizer: Normalizer, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "res_normalizer.pt"
    emb_path = out_dir / "embed_normalizer.pt"
    torch.save({"mean": res_normalizer.mean, "std": res_normalizer.std}, res_path)
    torch.save({"mean": embed_normalizer.mean, "std": embed_normalizer.std}, emb_path)
    return str(res_path), str(emb_path)

def _log_epoch_chunk_selection(
    tracker: "Tracker",
    epoch_idx: int,
    epoch_chunks: List[int],
    out_dir: Path,
    run_name: str = "probe",
):
    """
    Log the list of chunk IDs used in this epoch to W&B and upload as a text artifact.
    """
    preview = [int(c) for c in epoch_chunks[:16]]
    tracker.log({
        "epoch": epoch_idx + 1,
        "epoch/chunks_count": len(epoch_chunks),
        "epoch/chunks_preview_first16": preview,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{run_name}_epoch{epoch_idx+1:03d}_chunks.txt"
    with open(fname, "w") as f:
        f.write("\n".join(str(int(c)) for c in epoch_chunks))
    tracker.log_artifact(
        str(fname),
        name=f"epoch-{epoch_idx+1:03d}-chunks.txt",
        type_="metadata",
        description=f"Chunk IDs used for epoch {epoch_idx+1}",
    )

# ---------------------- entrypoint ------------------------------
def train_probe(
    start_chunk: int = 0,
    end_chunk: int = 0,
    n_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-6,
    lr_decay: float = 0.8,
    checkpoint_dir: str = str(Path(RESULTS_DIR) / "probes"),
    tracker: Optional[Tracker] = None,
    norm_chunks: int = 10,
    train_frac: float = 0.9,
    max_cached_chunks: int = 2,
    hf_repo_residuals: str = HF_REPO_ID_RESIDUALS,
    hf_repo_embeds: str = HF_REPO_ID_EMBEDS,
    hf_token: Optional[str] = os.environ.get("HF_TOKEN"),
    resume_from: Optional[str] = None,
    chunks_per_epoch: Optional[int] = None,
    chunk_seed: int = 42,  # kept for API compatibility; unused now that we don't shuffle
    eval_every: int = 1000,  # intermittent quick-eval cadence (steps)
    hf_cache_clean_gb: float = 15.0,
    hf_cache_clean_after_norm: bool = False,
    limit_layers: Optional[int] = None,
    local_residuals_dir: Optional[str] = None,
    local_embeds_dir: Optional[str] = None,
    prefer_local: bool = False,
):
    """Main training function (HF per-chunk streaming; ordered; optional contiguous slices)."""
    tracker = tracker or NoOpTracker()
    checkpoint_dir = Path(checkpoint_dir); checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Build full chunk range (ascending order)
    all_chunks = list(range(start_chunk, end_chunk + 1))
    assert len(all_chunks) > 0, "No chunks selected."
    disp = [f"{i:03d}" for i in all_chunks[:10]]
    if len(all_chunks) > 10:
        disp.append("...")
    print("Selected chunks (full range):", disp)

    # ------------------ Epoch selection (ordered, no shuffle) ------------------
    if chunks_per_epoch and chunks_per_epoch > 0:
        perm = all_chunks[:]  # already ordered
        epoch_slices: List[List[int]] = [perm[i:i + chunks_per_epoch] for i in range(0, len(perm), chunks_per_epoch)]
    else:
        epoch_slices = [all_chunks]  # full set each epoch

    # Normalization on a subset of the *full* range (first norm_chunks from full order)
    norm_chunk_ids = all_chunks[:min(norm_chunks, len(all_chunks))]
    res_norm, emb_norm = compute_normalizers(
        norm_chunk_ids,
        hf_repo_residuals, hf_repo_embeds, hf_token,
        local_residuals_dir, local_embeds_dir, prefer_local
    )

    # Optional: clean cache right after normalization
    # if hf_cache_clean_after_norm:
    #     maybe_clean_hf_cache(hf_cache_clean_gb, tracker=tracker)

    # Save normalizers locally (log via tracker)
    res_p, emb_p = _save_normalizers(res_norm, emb_norm, Path(RESULTS_DIR) / "normalizers")

    # Prepare a tiny temp dataset to infer shapes (use first slice)
    first_slice = epoch_slices[0]
    assert len(first_slice) > 0, "Empty epoch slice."
    _tmp_ds = StreamingOutlineDataset(
        hf_repo_residuals=hf_repo_residuals,
        hf_repo_embeds=hf_repo_embeds,
        chunk_ids=first_slice,
        split="train",
        train_frac=train_frac,
        res_normalizer=res_norm,
        embed_normalizer=emb_norm,
        max_cached_chunks=max_cached_chunks,
        hf_token=hf_token,
        limit_layers=limit_layers,
        local_residuals_dir=local_residuals_dir,
        local_embeds_dir=local_embeds_dir,
        prefer_local=prefer_local,
    )
    sample_res, _, _ = _tmp_ds[0]
    n_layers, d_model = sample_res.shape[1], sample_res.shape[2]
    print(f"\nInitializing probe:\n- Input: [batch, {n_layers} layers, {d_model} dims]\n- Output: 1024")
    probe = LinearProbe(n_layers=n_layers, d_model=d_model).to(DEVICE)

    # Build run config & init tracker BEFORE logging artifacts
    run_config = {
        "n_epochs": n_epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "weight_decay": weight_decay, "lr_decay": lr_decay,
        "start_chunk": start_chunk, "end_chunk": end_chunk,
        "device": str(DEVICE), "dtype": str(DTYPE),
        "hf_repo_residuals": hf_repo_residuals, "hf_repo_embeds": hf_repo_embeds,
        "n_layers": int(n_layers), "d_model": int(d_model), "d_sonar": 1024,
        "norm_chunks": norm_chunks, "norm_chunk_ids": norm_chunk_ids,
        "train_frac": train_frac, "max_cached_chunks": max_cached_chunks,
        "total_samples_full_range": len(all_chunks) * SAMPLES_PER_CHUNK,
        "chunks_per_epoch": chunks_per_epoch, "chunk_seed": chunk_seed,
        "resume_from": resume_from, "results_dir": str(RESULTS_DIR),
        "checkpoint_dir": str(checkpoint_dir),
        "hf_cache_clean_gb": hf_cache_clean_gb,
        "hf_cache_clean_after_norm": bool(hf_cache_clean_after_norm),
        "limit_layers": limit_layers,
        "local_residuals_dir": local_residuals_dir,
        "local_embeds_dir": local_embeds_dir,
        "prefer_local": prefer_local,
    }
    tracker.on_start(run_config); tracker.define_metrics(); tracker.watch(probe)

    # Optional: visible norm stats
    try:
        tracker.log({
            "norm/res_mean_abs": float(res_norm.mean.abs().mean().item()),
            "norm/res_std_mean": float(res_norm.std.mean().item()),
            "norm/emb_mean_abs": float(emb_norm.mean.abs().mean().item()),
            "norm/emb_std_mean": float(emb_norm.std.mean().item()),
        })
    except Exception:
        pass

    # Upload normalizer artifacts
    tracker.log_artifact(res_p, name="res_normalizer.pt", type_="asset", description="Residual normalizer")
    tracker.log_artifact(emb_p, name="embed_normalizer.pt", type_="asset", description="Embedding normalizer")

    # Tiny fixed audit set: last 5 chunks (or fewer if range small), use VAL split
    last_k = min(5, len(all_chunks))
    audit_chunks = all_chunks[-last_k:]
    print(f"Audit chunks (val split): {[f'{c:03d}' for c in audit_chunks]}")
    audit_val_dataset = StreamingOutlineDataset(
        hf_repo_residuals=hf_repo_residuals,
        hf_repo_embeds=hf_repo_embeds,
        chunk_ids=audit_chunks,
        split="val",
        train_frac=train_frac,
        res_normalizer=res_norm,
        embed_normalizer=emb_norm,
        max_cached_chunks=max_cached_chunks,
        hf_token=hf_token,
        limit_layers=limit_layers,
        local_residuals_dir=local_residuals_dir,
        local_embeds_dir=local_embeds_dir,
        prefer_local=prefer_local,
    )
    audit_loader = DataLoader(
        audit_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Trainer
    trainer = ProbeTrainer(
        probe=probe,
        lr=learning_rate,
        weight_decay=weight_decay,
        lr_decay=lr_decay,
        batch_size=batch_size,
        checkpoint_dir=str(checkpoint_dir),
        tracker=tracker,
    )

    # ------- Resume support -------
    start_epoch = 0
    if resume_from:
        if os.path.isfile(resume_from):
            ckpt = torch.load(resume_from, map_location=DEVICE)
            try:
                trainer.probe.load_state_dict(ckpt["model_state_dict"])
                trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                start_epoch = int(ckpt.get("epoch", 0) + 1)  # continue with NEXT epoch
                print(f"[resume] Loaded checkpoint '{resume_from}', resuming at epoch {start_epoch}.")
            except Exception as e:
                print(f"[resume] Failed to load checkpoint '{resume_from}': {e}")
        else:
            print(f"[resume] Checkpoint not found: {resume_from}")

    # ------- Epoch-wise training -------
    total_epochs = n_epochs
    for epoch in range(start_epoch, total_epochs):
        slice_idx = epoch % len(epoch_slices)
        epoch_chunks = epoch_slices[slice_idx]
        _log_epoch_chunk_selection(
            tracker=trainer.tracker,
            epoch_idx=epoch,
            epoch_chunks=epoch_chunks,
            out_dir=Path(RESULTS_DIR) / "chunk_schedules",
            run_name="probe",
        )
        disp = [f"{c:03d}" for c in epoch_chunks[:8]]
        if len(epoch_chunks) > 8:
            disp.append("...")
        print(f"\n[epoch {epoch+1}] Using {len(epoch_chunks)} chunks: {disp}")

        train_dataset = StreamingOutlineDataset(
            hf_repo_residuals=hf_repo_residuals,
            hf_repo_embeds=hf_repo_embeds,
            chunk_ids=epoch_chunks,
            split="train",
            train_frac=train_frac,
            res_normalizer=res_norm,
            embed_normalizer=emb_norm,
            max_cached_chunks=max_cached_chunks,
            hf_token=hf_token,
            limit_layers=limit_layers,
            local_residuals_dir=local_residuals_dir,
            local_embeds_dir=local_embeds_dir,
            prefer_local=prefer_local,
        )
        val_dataset = StreamingOutlineDataset(
            hf_repo_residuals=hf_repo_residuals,
            hf_repo_embeds=hf_repo_embeds,
            chunk_ids=epoch_chunks,
            split="val",
            train_frac=train_frac,
            res_normalizer=res_norm,
            embed_normalizer=emb_norm,
            max_cached_chunks=max_cached_chunks,
            hf_token=hf_token,
            limit_layers=limit_layers,
            local_residuals_dir=local_residuals_dir,
            local_embeds_dir=local_embeds_dir,
            prefer_local=prefer_local,
        )

        # Ordered, streaming loaders (low RAM)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
        )
        val_loader   = DataLoader(
            val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
        )

        # one epoch
        tr = trainer.train_epoch(epoch, train_loader, audit_loader=audit_loader, eval_every=eval_every)
        vl = trainer.validate(val_loader)

        print(f"\nEpoch {epoch+1} Summary:\n- Train Loss: {tr:.4f}\n- Val Loss: {vl:.4f}"
              f"\n- Learning Rate: {trainer.scheduler.get_last_lr()[0]:.2e}")
        trainer.tracker.log({"epoch": epoch + 1, "epoch/train_loss": tr, "epoch/val_loss": vl,
                             "epoch/lr": trainer.scheduler.get_last_lr()[0]})

        # checkpoint per epoch
        ckpt_path = os.path.join(str(checkpoint_dir), f"probe_epoch_{epoch+1}.pt")
        trainer.save_checkpoint(ckpt_path, epoch, tr, vl)
        logger.info(f"Saved checkpoint to {ckpt_path}")
        trainer.tracker.log_artifact(ckpt_path, name=f"checkpoint-epoch-{epoch+1}",
                                     type_="model", description=f"Linear probe after epoch {epoch+1}")

        # Clean HF cache if too big
        maybe_clean_hf_cache(hf_cache_clean_gb, tracker=trainer.tracker)

        trainer.scheduler.step()

    tracker.finish()
    return trainer, {"note": "ordered streaming with intermittent quick evals + HF/local fallback + HF cache autoclean"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Core probe trainer (HF/local streaming; ordered; intermittent evals).")
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--end-chunk", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--lr-decay", type=float, default=0.8)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--train-frac", type=float, default=0.9,
                        help="Global train fraction (0..1). Default 0.9 → 90% train / 10% val.")
    parser.add_argument("--norm-chunks", type=int, default=10,
                        help="Number of initial chunks to use for normalization stats (default: 10).")
    parser.add_argument("--max-cached-chunks", type=int, default=2,
                        help="LRU cache size in chunks (default: 2).")
    parser.add_argument("--checkpoint-dir", type=str, default=str(Path(RESULTS_DIR) / "probes"))
    parser.add_argument("--hf-repo-residuals", type=str, default=HF_REPO_ID_RESIDUALS)
    parser.add_argument("--hf-repo-embeds", type=str, default=HF_REPO_ID_EMBEDS)
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to a checkpoint .pt to resume training from.")
    parser.add_argument("--chunks-per-epoch", type=int, default=None,
                        help="If set, train/val only on this many chunks per epoch, rotating through the range (ordered, no shuffle).")
    parser.add_argument("--chunk-seed", type=int, default=42,
                        help="(Kept for compatibility) Seed ignored when not shuffling chunks.")
    parser.add_argument("--eval-every", type=int, default=1000,
                        help="Run a quick audit eval every N steps during training (default: 1000).")
    parser.add_argument("--hf-cache-clean-gb", type=float, default=15.0,
                        help="If HF cache exceeds this many GB, wipe it after each epoch.")
    parser.add_argument("--hf-cache-clean-after-norm", action="store_true",
                        help="Also clean HF cache immediately after the normalization pass.")
    parser.add_argument("--limit-layers", type=int, default=None,
                        help="If set, slice residuals to the first N layers before feeding the model.")
    # NEW: local fallbacks
    parser.add_argument("--local-residuals-dir", type=str, default=None,
                        help="Local directory containing res_data_XXX.pt residual chunks.")
    parser.add_argument("--local-embeds-dir", type=str, default=None,
                        help="Local directory containing outlines_XXX.pt embedding chunks.")
    parser.add_argument("--prefer-local", action="store_true",
                        help="Try local files first, then HF. Without this, tries HF first, then local.")

    args = parser.parse_args()

    if args.cpu:
        DEVICE = torch.device("cpu")

    train_probe(
        start_chunk=args.start_chunk,
        end_chunk=args.end_chunk,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_decay=args.lr_decay,
        n_epochs=args.n_epochs,
        train_frac=args.train_frac,
        norm_chunks=args.norm_chunks,
        max_cached_chunks=args.max_cached_chunks,
        checkpoint_dir=args.checkpoint_dir,
        hf_repo_residuals=args.hf_repo_residuals,
        hf_repo_embeds=args.hf_repo_embeds,
        hf_token=args.hf_token,
        resume_from=args.resume_from,
        chunks_per_epoch=args.chunks_per_epoch,
        chunk_seed=args.chunk_seed,
        eval_every=args.eval_every,
        hf_cache_clean_gb=args.hf_cache_clean_gb,
        hf_cache_clean_after_norm=args.hf_cache_clean_after_norm,
        limit_layers=args.limit_layers,
        local_residuals_dir=args.local_residuals_dir,
        local_embeds_dir=args.local_embeds_dir,
        prefer_local=args.prefer_local,
    )
