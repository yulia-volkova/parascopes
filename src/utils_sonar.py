# %%
from sonar.models.sonar_text import (
    load_sonar_text_decoder_model,
    load_sonar_text_encoder_model,
    load_sonar_tokenizer,
)
import torch
import torch.nn.functional as F
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import PaddingMask
from fairseq2.data.text import TextTokenDecoder

def load_sonar_text_encoder_model(encoder, device, dtype, progress=False):
    if isinstance(encoder, str):
        encoder = load_sonar_text_encoder_model(
            encoder, device=device, dtype=dtype, progress=False
        )
    return encoder

def load_sonar_text_decoder_model(decoder, device, dtype, progress=False):
    if isinstance(decoder, str):
        decoder = load_sonar_text_decoder_model(
            decoder, device=device, dtype=dtype, progress=False
        )
    return decoder

def load_sonar_tokenizer(tokenizer, progress=False):
    if isinstance(tokenizer, str):
        tokenizer = load_sonar_tokenizer(tokenizer, progress=False)
    return tokenizer

def get_ce_loss_decoder(input_embed, expected_text, decoder="text_sonar_basic_decoder",
                       tokenizer="text_sonar_basic_encoder", target_lang="eng_Latn",
                       device="cpu", dtype=None):
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
    # Load the model
    model = EmbeddingToTextModelPipeline(
        decoder=decoder,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype
    )

    # Tokenize the expected text
    tokenizer_decoder = model.tokenizer.create_decoder(lang=target_lang)
    tokenizer_encoder = model.tokenizer.create_encoder(lang=target_lang, device=device)
    expected_tokens = [tokenizer_encoder(text) for text in expected_text]

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
    padding_mask = PaddingMask(torch.stack(padding_mask_list))

    # Ensure input_embed is properly shaped [batch_size, 1, embedding_dim]
    if input_embed.dim() == 2:
        input_embed = input_embed.unsqueeze(1)

    # Forward pass through decoder model components
    with torch.no_grad():
        # Create a dummy batch for encoder output
        encoder_output = input_embed
        encoder_padding_mask = None

        # Get decoder frontend features
        target_tokens_shifted = target_tokens[:, :-1]  # Remove last token
        target_padding_mask_shifted = padding_mask.shift(-1)  # Shift padding mask

        seqs, padding_mask_dec = model.model.decoder.decoder_frontend(
            target_tokens_shifted, target_padding_mask_shifted)

        # Run through decoder
        decoder_output, _ = model.model.decoder.decoder(
            seqs, padding_mask_dec, encoder_output, encoder_padding_mask)

        # Get logits by projecting
        logits = model.model.project(decoder_output, padding_mask_dec).logits

    # Calculate cross-entropy loss
    target_tokens_shifted = target_tokens[:, 1:]  # Remove first token (BOS)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_tokens_shifted.reshape(-1),
        ignore_index=model.tokenizer.vocab_info.pad_idx
    )

    return loss

if __name__ == "__main__":
    example_text = "Greetings from the Sonar team!"
    input_embed = torch.randn(1, 1, 1024)
    loss = get_ce_loss_decoder(input_embed, [example_text], device="cpu")
    print(loss)
# %%
