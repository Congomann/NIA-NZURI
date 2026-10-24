import torch
import torch.nn as nn

class SwahiliS2STransformer(nn.Module):
    """
    Speech-to-Speech Transformer network.
    Takes multi-codebook discrete audio tokens for source speech and target context,
    maps them through embeddings and a transformer encoder-decoder, and projects
    them to logits across each codebook stream.
    """
    def __init__(
        self,
        num_tokens: int = 1024,
        num_codebooks: int = 4,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_codebooks = num_codebooks
        self.d_model = d_model

        # Codebook embedding layers
        self.src_embeddings = nn.ModuleList([
            nn.Embedding(num_tokens, d_model // num_codebooks) for _ in range(num_codebooks)
        ])
        self.tgt_embeddings = nn.ModuleList([
            nn.Embedding(num_tokens, d_model // num_codebooks) for _ in range(num_codebooks)
        ])

        # Core Transformer Encoder-Decoder Architecture
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Output projection heads for each discrete codebook stream
        self.heads = nn.ModuleList([
            nn.Linear(d_model, num_tokens) for _ in range(num_codebooks)
        ])

    def _embed_tokens(self, tokens: torch.Tensor, embeddings: nn.ModuleList) -> torch.Tensor:
        # tokens shape: [Batch, SeqLen, NumCodebooks]
        embedded_parts = [
            embeddings[i](tokens[:, :, i]) for i in range(self.num_codebooks)
        ]
        return torch.cat(embedded_parts, dim=-1)

    def forward(self, src_tokens: torch.Tensor, tgt_input: torch.Tensor):
        src = self._embed_tokens(src_tokens, self.src_embeddings)
        tgt = self._embed_tokens(tgt_input, self.tgt_embeddings)

        tgt_seq_len = tgt.shape[1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device=tgt.device)

        hidden_states = self.transformer(src, tgt, tgt_mask=tgt_mask)

        # Output: list of 4 tensor matrices representing each codebook stream track
        logits_list = [head(hidden_states) for head in self.heads]
        return logits_list


if __name__ == "__main__":
    print("Testing Swahili S2S Transformer Architecture...")
    model = SwahiliS2STransformer(num_tokens=1024, num_codebooks=4)
    
    # Simulate a mini-batch with 60 source tokens and 60 target tokens
    sim_src = torch.randint(0, 1024, (2, 60, 4))
    sim_tgt = torch.randint(0, 1024, (2, 59, 4))
    
    logits = model(sim_src, sim_tgt)
    print(f"Number of codebook output heads: {len(logits)}")
    print(f"Logits shape per head: {logits[0].shape}")
    print("Success! Transformer forward pass verified.")
