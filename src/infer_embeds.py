import os
from argparse import ArgumentParser
import torch
from utils_load_data import load_res_data  # Assumption: This returns a torch.Tensor or similar.
from utils_train import Trainer         # Assumption: Trainer class handles model loading & normalization.

NUM_COPIES = 1
#PREFIX = "sonar-sweeps"
PREFIX  = "notebooks-sonar"
POSTFIX = "_98"
INDEX = 98


def main():
    parser = ArgumentParser(description='Load MLP/Linear model and infer embeds from res_data')
    parser.add_argument('wandb_run_name', type=str,
                        help='Name of the W&B run to load (e.g., northern-sweep-37)')
    args = parser.parse_args()


    # Load the trainer, which wraps the model (assumed to be a linear/MLP model)
    trainer = Trainer.load_from_wandb(PREFIX + "/" + args.wandb_run_name)
    trainer.model.eval()
    DEVICE = trainer.device
    model_type = trainer.c.model_type

    # Load the full res_data 
    res_data = load_res_data(INDEX, groups_to_load=trainer.c.groups_to_load).to(DEVICE)

    # Normalize the res_data (assuming trainer.normalizer_res supports batched data)
    normalized_data = trainer.normalizer_res(res_data)

    # Run inference through the model and restore normalization on the output embeds
    with torch.no_grad():
        # For inference, we simply send the entire batch (NUM_COPIES replication is not needed here)
        predicted = trainer.model(normalized_data)
        predicted_embeds = trainer.normalizer_emb.restore(predicted)

    # Setup the output file path and check whether it exists.
    output_dir = "inferred_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"inferred_embeds_{args.wandb_run_name}{POSTFIX}_{model_type}.pt")
    if os.path.exists(output_path):
        print(f"Inferred embeds file already exists at: {output_path}. Skipping inference.")
        return

    # Save the inferred embeds output
    torch.save(predicted_embeds, output_path)
    print(f"Inferred embeds saved to: {output_path}")

if __name__ == "__main__":
    main()
