"""
wandb_tracker.py
A lightweight Weights & Biases implementation of the Tracker protocol used by train_probe.py,
PLUS a small CLI so you can launch training directly with W&B.

Examples:
  # login first: wandb login  (or export WANDB_API_KEY=...)
  python wandb_tracker.py \
    --residuals-path /workspace/hdd_cache/tensors/llama-3b \
    --hf-repo-id yulia-volkova/llama-3b-outlines-embeddings_new \
    --start-chunk 0 --end-chunk 19 \
    --n-epochs 5 --batch-size 32 --learning-rate 1e-4 \
    --norm-chunks 10 \
    --wandb-project probe-sonar \
    --wandb-entity yuulia-volkova-algoverse \
    --wandb-run-name llama3b-probe --wandb-resume allow
"""

from typing import Dict, Optional, List
import os
import argparse
import inspect

import torch
import torch.nn as nn

try:
    import wandb
except Exception as e:
    raise ImportError(
        "wandb is required for WandbTracker. Install with `pip install wandb`."
    ) from e

# --- import your core trainer (no wandb deps inside) ---
# We import the module to allow optional CPU forcing by tweaking its globals.
import train_probe as core


class WandbTracker:
    """
    Implements the Tracker protocol used by train_probe.py, backed by Weights & Biases.

    Methods:
        on_start(run_config: Dict)      -> initialize W&B run with config
        define_metrics()                -> declare step metrics for nice charts
        watch(model: nn.Module)         -> enable gradient/param logging
        log(data: Dict, step: int|None) -> log scalars/metrics
        log_artifact(path, name, type_, description) -> upload a file as an artifact
        finish()                        -> close the W&B run

    Notes:
        - Offline: set env `WANDB_MODE=offline`.
        - Resume: pass `--wandb-resume allow|must` and keep `WANDB_RUN_ID` stable
          (e.g., export WANDB_RUN_ID=my_probe_run).
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
        # Print the run id so you can reuse it for resume
        try:
            rid = self._run.id
            print(f"[wandb] Run ID: {rid}  (export WANDB_RUN_ID={rid} to resume)")
        except Exception:
            pass

    def define_metrics(self):
        """Declare metrics so W&B knows which field is the step axis."""
        wandb.define_metric("global_step", summary="max")
        wandb.define_metric("train/*", step_metric="global_step")
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
            wandb.log_artifact(art)  # fallback (rare)

    def finish(self):
        """Close the W&B run."""
        if self._run is not None:
            self._run.finish()
            self._run = None


# ----------------------------- CLI entrypoint -----------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the probe with Weights & Biases logging.")

    # Core training args (common)
    p.add_argument("--residuals-path", type=str, required=True)
    p.add_argument("--hf-repo-id", type=str, required=True)
    p.add_argument("--start-chunk", type=int, default=0)
    p.add_argument("--end-chunk", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--checkpoint-dir", type=str, default="results/probes")
    p.add_argument("--cpu", action="store_true")

    # Normalization stats subset
    p.add_argument("--norm-chunks", type=int, default=10,
                   help="Number of initial chunks to compute normalization stats on (default: 10).")

    # SPLIT / STREAMING OPTIONS (support both old and new cores):
    # - legacy per-chunk: train_size
    # - new global/streaming: train_frac + max_cached_chunks
    p.add_argument("--train-size", type=int, default=None,
                   help="LEGACY per-chunk train size (e.g., 800). If set, overrides train-frac.")
    p.add_argument("--train-frac", type=float, default=0.9,
                   help="GLOBAL train fraction (0..1). Used if core supports it and --train-size not set.")
    p.add_argument("--max-cached-chunks", type=int, default=2,
                   help="Streaming dataset LRU cache size in chunks (if core supports it).")

    # W&B args
    p.add_argument("--wandb-project", type=str, required=True, help="W&B project name")
    p.add_argument("--wandb-entity", type=str, default=None, help="W&B team/org (entity)")
    p.add_argument("--wandb-run-name", type=str, default=None, help="Run display name")
    p.add_argument("--wandb-tags", type=str, nargs="*", default=None, help="List of tags")
    p.add_argument("--wandb-resume", type=str, default=None, help='Resume mode: "allow" or "must"')

    return p.parse_args()


def _force_cpu_if_requested(wants_cpu: bool):
    """Best-effort way to ensure CPU only."""
    if not wants_cpu:
        return
    # 1) CUDA visibility off
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    # 2) Also try to poke the core’s DEVICE, if present
    try:
        if hasattr(core, "torch") and hasattr(core, "DEVICE"):
            core.DEVICE = core.torch.device("cpu")
            print("[wandb_tracker] Forced core.DEVICE to CPU")
    except Exception:
        pass


def _call_train_probe_filtered(args: argparse.Namespace, tracker: WandbTracker):
    """
    Call core.train_probe(...) but only with kwargs it actually accepts.
    This lets the same tracker work with both legacy and new core versions.
    """
    sig = inspect.signature(core.train_probe)
    accepted = set(sig.parameters.keys())

    # Base kwargs common to both versions
    kwargs = dict(
        residuals_path=args.residuals_path,
        hf_repo_id=args.hf_repo_id,
        start_chunk=args.start_chunk,
        end_chunk=args.end_chunk,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        checkpoint_dir=args.checkpoint_dir,
        tracker=tracker,
        norm_chunks=args.norm_chunks,
    )

    # Split/streaming flags: prefer legacy train_size if provided
    if args.train_size is not None:
        if "train_size" in accepted:
            kwargs["train_size"] = args.train_size
    else:
        # try to pass train_frac / max_cached_chunks if supported by core
        if "train_frac" in accepted:
            kwargs["train_frac"] = args.train_frac
        if "max_cached_chunks" in accepted:
            kwargs["max_cached_chunks"] = args.max_cached_chunks

    # Filter to only those actually accepted by the core
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    # Warn if something was dropped (optional)
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
