import os
import json  # for saving output as JSON
from argparse import ArgumentParser
import torch
from tqdm import tqdm
from utils_load_data import load_embeds, BASE_DIR

from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

def decode_file(embeds, vec2text_model, batch_size, device):
    """
    Loads a tensor from file_path, decodes its embeddings in batches,
    and returns the list of decoded texts.
    """
    # Load embeds and ensure they are on the right device.

    decoded_texts = []
    num_rows = embeds.size(0)
    for start_idx in tqdm(range(0, num_rows, batch_size)):
        batch = embeds[start_idx: start_idx + batch_size]
        # Predict returns a list of decoded texts according to the pipeline's implementation.
        decoded_batch = vec2text_model.predict(batch, target_lang="eng_Latn")
        decoded_texts.extend(decoded_batch)
    return decoded_texts

def main():
    parser = ArgumentParser(
        description='Decode SONAR embeddings from a specific inferred embed file'
    )
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for decoding (default: 32)')

    args = parser.parse_args()

    # Determine the device for computation.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the SONAR decoding pipeline.
    vec2text_model = EmbeddingToTextModelPipeline(
        decoder="text_sonar_basic_decoder",
        tokenizer="text_sonar_basic_encoder",
        device=device
    )
    vec2text_model = vec2text_model.to(torch.float16)

    torch.compile(vec2text_model)

    print("- using:")
    print(vec2text_model.device)
    print(vec2text_model.model.decoder.decoder_frontend.embed.weight.dtype)

    # Define the fixed file to load.
    #input_path  = f"{BASE_DIR}/inferred_outputs/inferred_embeds_iqzigl1h.pt"
    #output_path = f"{BASE_DIR}/comparison_texts/mlp_decoded_texts.json"
    #input_path = f"{BASE_DIR}/inferred_outputs/inferred_embeds_4nbwxrar_98_linear.pt"
    #output_path = f"{BASE_DIR}/comparison_texts/linear_train_decoded_texts.json"
    #input_path = f"{BASE_DIR}/inferred_outputs/inferred_embeds_iqzigl1h_98_mlp.pt"
    #output_path = f"{BASE_DIR}/comparison_texts/mlp_train_decoded_texts.json"
    input_path = f"{BASE_DIR}/inferred_outputs/inferred_embeds_fnjt2per_99_linear.pt"
    output_path = f"{BASE_DIR}/comparison_texts/linear_ce_decoded_texts.json"
    if not os.path.isfile(input_path):
        print(f"File not found: {input_path}")
        return
    embeds = torch.load(input_path).to(device, torch.float16)


    # embeds = load_embeds(99).to(device)
    # output_path = "comparison_texts/original_decoded_output.json"

    print(f"Decoding embeds: {embeds.shape}")
    decoded_texts = decode_file(embeds, vec2text_model, args.batch_size, device)

    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the decoded texts list to JSON.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(decoded_texts, f, indent=4)

    print(f"Decoded texts saved to: {output_path}")

if __name__ == "__main__":
    with torch.inference_mode():
        main()
