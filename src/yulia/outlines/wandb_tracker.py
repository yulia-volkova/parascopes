from typing import Dict, Optional, List
import os
import argparse
import inspect
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

try:
    import wandb
except Exception as e:
    raise ImportError(
        "wandb is required for WandbTracker. Install with `pip install wandb`."
    ) from e

# --- robustly import the sibling train_probe.py regardless of CWD ---
_here = Path(__file__).parent
_train_probe_path = _here / "train_probe.py"
if not _train_probe_path.exists():
    raise FileNotFoundError(f"Could not find train_probe.py at: {_train_probe_path}")

_spec = importlib.util.spec_from_file_location("train_probe", _train_probe_path)
core = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader, "Failed to build import spec for train_probe.py"
_spec.loader.exec_module(core)


class WandbTracker:
    """
    Implements the Tracker protocol used by train_probe.py, backed by Weights & Biases.
    """

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        resume: Optional[str] = None,  # "allow" | "must" | None
        watch_log: str = "gradients",  # or "parameters"
        watch_log_freq: int = 200,
    ):
        self._init_kwargs = dict(
            project=project,
            entity=entity,
            name=name,
            tags=tags or [],
            resume=resume or os.environ.get("WANDB_RESUME"),
        )
        self._watch_log = watch_log
        self._watch_log_freq = watch_log_freq
        self._run = None

    # ---- Tracker protocol methods ----

    def on_start(self, run_config: Dict):
        """Initialize the W&B run and set the configuration."""
        self._run = wandb.init(**self._init_kwargs, config=run_config)
        try:
            rid = self._run.id
            print(f"[wandb] Run ID: {rid}  (export WANDB_RUN_ID={rid} to resume)")
        except Exception:
            pass

    def define_metrics(self):
        """Declare metrics so W&B knows which field is the step axis."""
        # Global step is the x-axis for per-step logs
        wandb.define_metric("global_step", summary="max")

        # Per-step training metrics (already logged with 'global_step')
        wandb.define_metric("train/*", step_metric="global_step")

        # Intermittent quick evals during training use global_step too
        wandb.define_metric("quick/*", step_metric="global_step")

        # End-of-epoch summaries (val/* and epoch/*) should step on 'epoch'
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("epoch/*", step_metric="epoch")

    def watch(self, model: nn.Module):
        """Enable gradients/parameters logging."""
        wandb.watch(model, log=self._watch_log, log_freq=self._watch_log_freq)

    def log(self, data: Dict, step: Optional[int] = None):
        """Log a dictionary of scalars/metrics."""
        wandb.log(data, step=step)

    def log_artifact(
        self,
        path: str,
        name: Optional[str] = None,
        type_: str = "asset",
        description: str = "",
    ):
        """Upload a file as a W&B artifact."""
        art = wandb.Artifact(
            name=name or os.path.basename(path),
            type=type_,
            description=description,
        )
        art.add_file(path, name=name or os.path.basename(path))
        if self._run is not None:
            self._run.log_artifact(art)
        else:
            wandb.log_artifact(art)  # fallback

    def finish(self):
        """Close the W&B run."""
        if self._run is not None:
            self._run.finish()
            self._run = None


# ----------------------------- CLI entrypoint -----------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the probe with Weights & Biases logging.")

    # Core training args (common)
    p.add_argument("--start-chunk", type=int, default=0)
    p.add_argument("--end-chunk", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--lr-decay", type=float, default=0.8)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--checkpoint-dir", type=str, default="results/probes")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to a checkpoint .pt to resume training from.")
    p.add_argument("--cpu", action="store_true")

    # HF + streaming options
    p.add_argument("--hf-repo-residuals", type=str, default="nickypro/fineweb-llama3b-residuals")
    p.add_argument("--hf-repo-embeds", type=str, default="yulia-volkova/llama-3b-outlines-embeddings_new")
    p.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    p.add_argument("--chunks-per-epoch", type=int, default=None,
                   help="If set, use this many chunks per epoch (ordered slices, no shuffle).")
    p.add_argument("--chunk-seed", type=int, default=42,
                   help="Kept for compatibility; unused when not shuffling.")
    p.add_argument("--train-frac", type=float, default=0.9,
                   help="GLOBAL train fraction (0..1).")
    p.add_argument("--max-cached-chunks", type=int, default=2,
                   help="Streaming dataset LRU cache size in chunks.")
    p.add_argument("--norm-chunks", type=int, default=10,
                   help="Number of initial chunks to compute normalization stats on.")
    p.add_argument("--eval-every", type=int, default=1000,
                   help="Run a quick audit eval every N steps during training.")
    p.add_argument("--limit-layers", type=int, default=None,
                   help="If set, slice residuals to the first N layers before the probe.")

    # HF cache cleaning
    p.add_argument("--hf-cache-clean-gb", type=float, default=15.0,
                   help="If HF cache exceeds this many GB, wipe it after each epoch.")
    p.add_argument("--hf-cache-clean-after-norm", action="store_true",
                   help="Also clean HF cache immediately after the normalization pass.")

    # NEW: Local fallbacks
    p.add_argument("--local-residuals-dir", type=str, default=None,
                   help="Local directory containing res_data_XXX.pt residual chunks.")
    p.add_argument("--local-embeds-dir", type=str, default=None,
                   help="Local directory containing outlines_XXX.pt embedding chunks.")
    p.add_argument("--prefer-local", action="store_true",
                   help="Try local files first, then HF. Without this, tries HF first, then local.")

    # W&B args
    p.add_argument("--wandb-project", type=str, required=True, help="W&B project name")
    p.add_argument("--wandb-entity", type=str, default=None, help="W&B team/org (entity)")
    p.add_argument("--wandb-run-name", type=str, default=None, help="Run display name")
    p.add_argument("--wandb-tags", type=str, nargs="*", default=None, help="List of tags")
    p.add_argument("--wandb-resume", type=str, default=None, help='Resume mode: "allow" or "must"')

    # Legacy fields (kept so older cores still work)
    p.add_argument("--residuals-path", type=str, default=None)  # legacy, ignored by new core
    p.add_argument("--hf-repo-id", type=str, default=None)      # legacy, ignored by new core
    p.add_argument("--train-size", type=int, default=None,      # legacy per-chunk split
                   help="LEGACY per-chunk train size (e.g., 800). If set, overrides train-frac.")

    return p.parse_args()


def _force_cpu_if_requested(wants_cpu: bool):
    """Best-effort way to ensure CPU only."""
    if not wants_cpu:
        return
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    try:
        if hasattr(core, "torch") and hasattr(core, "DEVICE"):
            core.DEVICE = core.torch.device("cpu")
            print("[wandb_tracker] Forced core.DEVICE to CPU")
    except Exception:
        pass


def _call_train_probe_filtered(args: argparse.Namespace, tracker: WandbTracker):
    """
    Call core.train_probe(...) but only with kwargs it actually accepts.
    This lets the same tracker work with different core versions.
    """
    sig = inspect.signature(core.train_probe)
    accepted = set(sig.parameters.keys())

    kwargs = dict(
        # modern HF/streaming args
        hf_repo_residuals=args.hf_repo_residuals,
        hf_repo_embeds=args.hf_repo_embeds,
        hf_token=args.hf_token,

        start_chunk=args.start_chunk,
        end_chunk=args.end_chunk,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_decay=args.lr_decay,
        n_epochs=args.n_epochs,
        checkpoint_dir=args.checkpoint_dir,
        tracker=tracker,
        norm_chunks=args.norm_chunks,
        resume_from=args.resume_from,

        # streaming controls
        train_frac=args.train_frac,
        max_cached_chunks=args.max_cached_chunks,
        chunks_per_epoch=args.chunks_per_epoch,
        chunk_seed=args.chunk_seed,

        # intermittent quick eval cadence
        eval_every=args.eval_every,

        # HF cache clean controls
        hf_cache_clean_gb=args.hf_cache_clean_gb,
        hf_cache_clean_after_norm=args.hf_cache_clean_after_norm,

        # model/input shaping
        limit_layers=args.limit_layers,

        # NEW: local fallbacks
        local_residuals_dir=args.local_residuals_dir,
        local_embeds_dir=args.local_embeds_dir,
        prefer_local=args.prefer_local,

        # legacy (only if core expects them)
        residuals_path=args.residuals_path,
        hf_repo_id=args.hf_repo_id,
        train_size=args.train_size,
    )

    # Filter to only those the core actually accepts
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = [k for k in kwargs.keys() if k not in accepted]
    if dropped:
        print(f"[wandb_tracker] Note: core.train_probe() does not accept: {dropped}. Skipping those args.")

    return core.train_probe(**filtered_kwargs)


def main():
    args = _parse_args()
    _force_cpu_if_requested(args.cpu)

    tracker = WandbTracker(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=args.wandb_tags,
        resume=args.wandb_resume,
    )

    _call_train_probe_filtered(args, tracker)


if __name__ == "__main__":
    main()
