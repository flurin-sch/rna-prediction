import torch
from torch import nn


class ChromatinModel(nn.Module):
    """Direct measured-signal predictor, optionally fused with the sequence encoder."""
    def __init__(self, chromatin_dim, sequence_model=None, width=128, dropout=0.1):
        super().__init__()
        self.sequence_model = sequence_model
        self.chromatin = nn.Sequential(nn.Linear(chromatin_dim, width), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(width, width), nn.ReLU())
        self.direct = nn.Linear(width, 1)
        if sequence_model is not None:
            sequence_width = sequence_model.projection[1].out_features
            self.joint = nn.Sequential(nn.Linear(width + sequence_width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, chromatin, sequence=None, features=None, valid=None):
        measured = self.chromatin(chromatin)
        score = self.direct(measured).squeeze(-1)
        if self.sequence_model is not None:
            encoded = self.sequence_model.encode(sequence, features, valid)
            score = (score + self.sequence_model.predictor(encoded).squeeze(-1)
                     + self.joint(torch.cat([measured, encoded], dim=-1)).squeeze(-1))
        return score

