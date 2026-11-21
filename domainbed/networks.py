import torch
import torch.nn as nn


class SeqCNN(nn.Module):
    """The Antibody DomainBed sequence CNN: one embedding per chain type, two
    shared convolutions, max-pooling over positions, and a linear feature layer."""
    n_outputs = 256

    def __init__(self, seq_len, d_model=64, ksizes=(10, 5), stride=2):
        super().__init__()
        # unused; kept so that parameter initialization matches the reported runs
        self.embed = nn.Embedding(seq_len + 1, d_model, padding_idx=0)
        self.emb_ab = nn.Embedding(22, d_model, padding_idx=0)
        self.emb_ag = nn.Embedding(22, d_model, padding_idx=0)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=ksizes[0], stride=stride),
            nn.ReLU(),
            nn.Conv1d(d_model, 2 * d_model, kernel_size=ksizes[1], stride=stride),
            nn.ReLU(),
        )
        self.ffd = nn.Linear(4 * d_model, self.n_outputs)

    def forward(self, x):
        """x: one-hot [batch, 298 + antigen length, 22]; the first 298 positions are the antibody."""
        e_ab = self.emb_ab(x[:, :298].argmax(dim=2))
        e_ag = self.emb_ag(x[:, 298:].argmax(dim=2))
        h_ab = self.conv_layers(e_ab.permute(0, 2, 1))
        h_ag = self.conv_layers(e_ag.permute(0, 2, 1))
        return self.ffd(torch.cat([h_ab.max(dim=2)[0], h_ag.max(dim=2)[0]], dim=1))


def Featurizer(input_shape):
    return SeqCNN(input_shape[0])


def Classifier(in_features, out_features):
    return nn.Linear(in_features, out_features)
