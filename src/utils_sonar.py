# %%
from sonar.models.sonar_text import (
    load_sonar_text_decoder_model,
    load_sonar_text_encoder_model,
    load_sonar_tokenizer,
)
import torch
import torch.nn.functional as F
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import PaddingMask
from fairseq2.data.text import TextTokenDecoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text2vec = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder", device=DEVICE)
vec2text = EmbeddingToTextModelPipeline(decoder="text_sonar_basic_decoder", tokenizer="text_sonar_basic_encoder", device=DEVICE)
tokenizer = load_sonar_tokenizer("text_sonar_basic_encoder")

torch.set_grad_enabled(False)

# %%

def get_ce_loss_decoder(model, input_embed, expected_text, target_lang="eng_Latn", device=DEVICE, dtype=None):
    """
    Compute cross-entropy loss between decoder output and expected text.

    Args:
        input_embed: Tensor of shape [batch_size, embedding_dim] - SONAR embeddings
        expected_text: List of strings - the expected output text
        decoder: String or decoder model
        tokenizer: String or tokenizer
        target_lang: Target language code
        device: Device to run on
        dtype: Data type to use

    Returns:
        Cross-entropy loss
    """
    # Tokenize the expected text
    tokenizer_decoder = model.tokenizer.create_decoder()
    tokenizer_encoder = model.tokenizer.create_encoder(device=device)
    expected_tokens = [tokenizer_encoder(text) for text in expected_text]
    bos_tensor = torch.tensor([3], device=expected_tokens[0].device, dtype=expected_tokens[0].dtype)
    expected_tokens = [torch.cat([bos_tensor, tokens]) for tokens in expected_tokens]

    # Create padding mask and collate the expected tokens
    max_len = max(len(tokens) for tokens in expected_tokens)
    padded_tokens = []
    padding_mask_list = []

    for tokens in expected_tokens:
        pad_len = max_len - len(tokens)
        padded = torch.cat([tokens, torch.full((pad_len,), model.tokenizer.vocab_info.pad_idx,
                                              device=tokens.device, dtype=tokens.dtype)])
        padded_tokens.append(padded)
        padding_mask_list.append(torch.cat([torch.ones(len(tokens), dtype=torch.bool, device=tokens.device),
                                           torch.zeros(pad_len, dtype=torch.bool, device=tokens.device)]))

    target_tokens = torch.stack(padded_tokens).to(device)
    stacked_padding_mask = torch.stack(padding_mask_list)

    # Create PaddingMask with both the mask and sequence length
    seq_lens = torch.sum(stacked_padding_mask, dim=1).to(torch.int)
    padding_mask = PaddingMask(stacked_padding_mask, seq_lens)

    # Ensure input_embed is properly shaped [batch_size, 1, embedding_dim]
    if input_embed.dim() == 2:
        input_embed = input_embed.unsqueeze(1)

    # Split into input tokens (all but last) and target tokens (all but first)
    input_tokens = target_tokens[:, :-1]  # Remove last token
    target_labels = target_tokens[:, 1:]  # Remove first token (BOS)

    # Create appropriate padding masks
    input_seq_lens = torch.clamp(seq_lens - 1, min=0)
    input_padding_mask = PaddingMask(input_seq_lens, batch_seq_len=max_len - 1)

    # Forward pass through model components
    with torch.no_grad():
        # Prepare encoder output (our embedding)
        encoder_output = input_embed
        encoder_padding_mask = None

        print("Input tokens: ", [(x.item(), tokenizer_decoder(x[None])) for x in input_tokens.flatten()])

        # Get decoder frontend features (token embeddings + positional encoding)
        seqs, padding_mask_dec = model.model.decoder.decoder_frontend(
            input_tokens, input_padding_mask)

        print({"Seqs": seqs.shape})
        # print("Seqs: ", [(x.item(), tokenizer_decoder(x[None])) for x in seqs.flatten()])

        # Run through decoder
        decoder_output, _ = model.model.decoder.decoder(
            seqs.to(device), padding_mask_dec.to(device), encoder_output.to(device), encoder_padding_mask)

        print(decoder_output.shape)

        # Get logits by projecting
        logits = model.model.project(decoder_output, padding_mask_dec).logits

    # Calculate cross-entropy loss
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_labels.reshape(-1),
        ignore_index=model.tokenizer.vocab_info.pad_idx
    )

    # Calculate accuracy and token probabilities
    probs = F.softmax(logits, dim=-1)

    # Track per-token probabilities and predictions
    batch_size, seq_len, vocab_size = logits.shape
    token_probs = []
    correct_count = 0
    total_count = 0

    print("\nTOKEN PREDICTIONS:")
    vocab = model.tokenizer.vocab_info

    for b in range(batch_size):
        for s in range(seq_len):
            # Skip padding tokens
            if s >= input_seq_lens[b]:
                continue

            # Get the predicted and target token
            pred_idx = torch.argmax(logits[b, s]).item()
            target_idx = target_labels[b, s].item()

            # Skip padding in target
            if target_idx == vocab.pad_idx:
                continue

            # Track statistics
            total_count += 1
            correct = (pred_idx == target_idx)
            if correct:
                correct_count += 1

            # Get probability of the target token
            target_prob = probs[b, s, target_idx].item()
            token_probs.append(target_prob)

            # Print token information
            prefix = "✓" if correct else "✗"
            dec = lambda x: tokenizer_decoder(torch.tensor([x]))
            print(f"{prefix} Position {s+1}: Target prob={target_prob:.4f}, Predicted: {pred_idx} '{dec(pred_idx)}', Expected: {target_idx} '{dec(target_idx)}'")

    # Print summary statistics
    accuracy = correct_count / total_count if total_count > 0 else 0
    avg_prob = sum(token_probs) / len(token_probs) if token_probs else 0

    print(f"\nSUMMARY:")
    print(f"Accuracy: {accuracy:.2f} ({correct_count}/{total_count})")
    print(f"Average target token probability: {avg_prob:.4f}")
    print(f"Loss: {loss.item():.4f}")

    return loss

if __name__ == "__main__":
    example_text = "Greetings from the Sonar team!"
    input_embed = torch.randn(1, 1, 1024)
    loss = get_ce_loss_decoder(vec2text, input_embed, [example_text], device=DEVICE)
    print(loss)
    input_embed = text2vec.predict([example_text], source_lang="eng_Latn")
    output_text = vec2text.predict(input_embed, target_lang="eng_Latn")
    print(output_text)

    loss = get_ce_loss_decoder(vec2text, input_embed, [example_text], device=DEVICE)
    print(loss)
# %%
