# %% load up a model
import textwrap
from termcolor import colored
from argparse import ArgumentParser
import torch
from utils_load_data import load_embeds, load_res_data, load_res_data_layer
from utils_train import Trainer  # Assumption: You have a Trainer class with loading functionality
from utils_train_layer import Trainer as TrainerLayer

NUM_COPIES = 1
#PREFIX = "sonar-sweeps"
PREFIX = "notebooks-sonar"

def main():
    parser = ArgumentParser(description='Test the model')
    parser.add_argument('wandb_run_name', type=str,
                       help='Name of the W&B run to load (e.g., northern-sweep-37)')
    parser.add_argument('-l', '--layer', action='store_true', default=False)
    args = parser.parse_args()
    if args.layer:
        model = TrainerLayer.load_from_wandb(PREFIX+"/"+args.wandb_run_name)
    else:
        model = Trainer.load_from_wandb(PREFIX+"/"+args.wandb_run_name)
    return model

if __name__ == "__main__":
    trainer = main()
    trainer.model.eval()
    DEVICE = trainer.device

    print(f"Testing some outputs, running on {DEVICE}")
    from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
    vec2text_model = EmbeddingToTextModelPipeline(decoder="text_sonar_basic_decoder", tokenizer="text_sonar_basic_encoder", device=DEVICE)

    # Example usage with the vec2text_model
    with torch.no_grad():
        embeds = load_embeds(99)
        if hasattr(trainer.c, 'chosen_layer'):
            res_data = load_res_data_layer(99, trainer.c.chosen_layer, group_size=trainer.c.group_size, group_operation=trainer.c.group_operation)
        else:
            res_data = load_res_data(99, group_size=trainer.c.group_size, groups_to_load=trainer.c.groups_to_load, group_operation=trainer.c.group_operation)
        for index in [1, 42, 100, 200, 300, 500, 800, 1000, 1234, 2345, 3456]:
            prev_emb = embeds[index-1].unsqueeze(dim=0).to(DEVICE)
            orig_emb   = embeds[index].unsqueeze(dim=0).to(DEVICE)
            test_input = trainer.normalizer_res(res_data[index].unsqueeze(dim=0).to(DEVICE))
            test_input_batch = test_input.repeat(NUM_COPIES, 1)  # Create a batch of 3 copies along dim=0
            predicted_embeds = trainer.normalizer_emb.restore(trainer.model(test_input_batch))
            decoded_text = vec2text_model.predict(
                torch.cat([prev_emb, orig_emb, predicted_embeds], dim=0),
                target_lang="eng_Latn"
            )
            cossim = torch.nn.functional.cosine_similarity(orig_emb[0], predicted_embeds[0], dim=0).item()
            print(f"### EXAMPLE {index} ### (SIM = {cossim:.4f})")

            print(textwrap.fill(colored(f"PREV    : {decoded_text[0][:200]}", 'blue'),
                              width=120,
                              initial_indent='',
                              subsequent_indent=' ' * 10))
            print(textwrap.fill(colored(f"ORIGINAL: {decoded_text[1][:200]}", 'blue'),
                              width=120,
                              initial_indent='',
                              subsequent_indent=' ' * 10))
            for i in range(2, len(decoded_text)):
                print(textwrap.fill(colored(f"PROBE   : {decoded_text[i][:200]}", 'green'),
                                    width=120,
                                    initial_indent='',
                                    subsequent_indent=' ' * 10))
            print()

# %%
