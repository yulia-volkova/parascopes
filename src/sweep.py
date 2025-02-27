# %% Imports
import os
import gc
import torch
import wandb
from utils_train import Trainer
from utils_load_data import load_res_data, load_embeds, load_paragraphs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# %% Sweep Configuration
k = 1024
sweep_config = {
    'method': 'random',  # Options: grid, random, bayes
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'model_type': {
            'values': ['mlp']
        },
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 5e-6,  # Actual value range
            'max': 2e-5
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-7,
            'max': 1e-4
        },
        'batch_size': {
            'values': [32, 64, 128, 256, 512, 1024]
        },
        'groups_to_load': {
            'values': [1,2,3,4,5,6,7,8,9,10,11,12]
        },
        'dropout': {
            'values': [0.0, 0.05, 0.1, 0.2]
        },
        'd_mlp': {
            'values': [k, 2*k, 4*k, 8*k, 12*k, 16*k ]
        },
    },
}

# %% Core Training Function
def train_sweep():
    try:
        # Initialize WandB with sweep config
        wandb.init()

        # Build config from sweep parameters
        c = wandb.config
        config = {
            'model_type': c.model_type,
            'batch_size': c.batch_size,
            'group_size': 4,
            'groups_to_load': c.groups_to_load,
            'lr': c.lr,
            'weight_decay': c.weight_decay,
            'dropout': c.dropout,
            'd_mlp': c.d_mlp,
            'num_epochs': 3,
            'num_files': 99,
            'd_sonar': 1024
        }

        # Create and run trainer
        trainer = Trainer(config, DEVICE)
        model = trainer.train()

        # Save model with sweep metadata
        filename = f"./checkpoints/sweeps/{wandb.run.id}_{wandb.config.model_type}.pkl"
        trainer.save_checkpoint(filename)
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print(f"CUDA OOM error occurred: {e}")
            if wandb.run:
                wandb.run.finish(exit_code=1)
            gc.collect()
            torch.cuda.empty_cache()
        else:
            raise

# %% Main Execution
if __name__ == "__main__":
    import sys
    import os
    import time
    import subprocess

    if len(sys.argv) > 1:
        # Child process - run agent with specified sweep_id
        sweep_id = sys.argv[1]
        gpu_id = sys.argv[2]
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            wandb.agent(sweep_id, function=train_sweep, project="sonar-sweeps")
        except Exception as e:
            print(f"Agent failed on GPU {gpu_id}: {e}")
    else:
        # Parent process - create sweep and launch subprocesses
        sweep_id = wandb.sweep(sweep_config, project="sonar-sweeps")
        print(f"Sweep ID: {sweep_id}")

        processes = []
        wandb_key = os.environ.get("WANDB_API_KEY", "")

        for gpu_idx in range(4):
            env = os.environ.copy()
            env["WANDB_API_KEY"] = wandb_key
            env["WANDB_DIR"] = f"./wandb_{gpu_idx}"  # Separate wandb directories

            process = subprocess.Popen(
                [sys.executable, __file__, sweep_id, str(gpu_idx)],
                env=env
            )
            processes.append(process)
            time.sleep(5)  # Longer delay between agent starts

        # Wait for all processes to complete
        for process in processes:
            process.wait()
