# %%
import os
import gc
import torch
import wandb

from utils_load_data import load_res_data, load_embeds, load_paragraphs
from utils_welford   import load_or_compute_welford_stats
from utils_train     import Trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

def train_linear():
    try:
        wandb.init(project="notebooks-sonar")

        config = {
            'model_type': 'mlp',
            'batch_size': 1024,
            'num_epochs': 10,
            'num_files': 99,
            'group_size': 4,
            'groups_to_load': 6,
            'lr': 2e-5,
            'lr_decay': 0.8,
            'weight_decay': 2e-5,
            'dropout': 0.05,
            'd_mlp': 8192,
            'd_sonar': 1024,
            # 'd_res': 61440
        }
        wandb.config.update(config)

        # Create and run trainer
        trainer = Trainer(config, DEVICE)
        model = trainer.train()

        # Save checkpoint with wandb metadata
        filename = f"./checkpoints/sweeps/{wandb.run.id}_{wandb.config.model_type}.pkl"
        trainer.save_checkpoint(filename)
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print(f"CUDA OOM error occurred: {e}")
            if wandb.run:
                wandb.run.finish(exit_code=1)
            gc.collect()
            torch.cuda.empty_cache()
            import traceback
            traceback.print_exc()

        else:
            import traceback
            traceback.print_exc()
            raise

if __name__ == "__main__":
    train_linear()

# %%
