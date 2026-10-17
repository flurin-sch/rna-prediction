import torch
from torch import nn


def promoter_attention_mask(regions, device=None):
    """True blocks an edge; promoter links and diagonal edges remain allowed."""
    mask = ~torch.eye(regions, dtype=torch.bool, device=device)
    mask[0, :] = False
    mask[:, 0] = False
    return mask


class InteractionBlock(nn.Module):
    def __init__(self, width, heads, feedforward):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.0)
        self.feedforward = nn.Sequential(nn.Linear(width, feedforward), nn.ReLU(), nn.Linear(feedforward, width))

    def forward(self, tokens, valid, mask):
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized,
                                     key_padding_mask=~valid, attn_mask=mask, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.feedforward(self.norm2(tokens))
        return tokens.masked_fill(~valid.unsqueeze(-1), 0)


class ExpressionModel(nn.Module):
    def __init__(self, feature_dim, embedding_dim=3072, width=64, heads=4, layers=3, feedforward=256):
        super().__init__()
        self.embedding_dim, self.feature_dim = embedding_dim, feature_dim
        self.projection = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, width))
        self.fusion = nn.Sequential(nn.Linear(width + feature_dim, width), nn.ReLU(),
                                    nn.Linear(width, width), nn.ReLU())
        self.blocks = nn.ModuleList([InteractionBlock(width, heads, feedforward) for _ in range(layers)])
        self.predictor = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Linear(128, 128),
                                       nn.ReLU(), nn.Linear(128, 1))

    def encode(self, sequence, features, valid):
        if sequence.ndim != 3 or sequence.shape[-1] != self.embedding_dim:
            raise ValueError("Invalid sequence embedding shape")
        if features.shape != (*sequence.shape[:2], self.feature_dim) or valid.shape != sequence.shape[:2]:
            raise ValueError("Feature/mask dimensions do not match sequence embeddings")
        if valid.dtype != torch.bool or not valid[:, 0].all():
            raise ValueError("Validity mask must be boolean and promoter must always be valid")
        # Sanitize before projection: even NaN/inf in padded input must be irrelevant.
        sequence = sequence.masked_fill(~valid.unsqueeze(-1), 0)
        features = features.masked_fill(~valid.unsqueeze(-1), 0)
        tokens = self.fusion(torch.cat([self.projection(sequence), features], dim=-1))
        tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0)
        mask = promoter_attention_mask(sequence.shape[1], sequence.device)
        for block in self.blocks:
            tokens = block(tokens, valid, mask)
        return tokens[:, 0]

    def forward(self, sequence, features, valid):
        return self.predictor(self.encode(sequence, features, valid)).squeeze(-1)

