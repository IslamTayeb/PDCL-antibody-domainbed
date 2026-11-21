import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

AMINO_ACIDS = ['M', 'N', 'P', 'Y', 'D', 'H', 'Q', 'C', 'K', 'G', 'R', 'S', 'E', 'I', 'L', 'T', 'F', 'V', 'A', 'W', '-', '.']


class SingleRound(Dataset):
    """One design round: one-hot heavy chain, light chain and padded antigen."""

    def __init__(self, df, max_target_len):
        self.df = df.reset_index(drop=True)
        self.max_target_len = max_target_len
        self.aa_to_int = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

    def __len__(self):
        return len(self.df)

    def _onehot(self, seq):
        return F.one_hot(torch.tensor([self.aa_to_int[r] for r in seq]), num_classes=22)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        heavy = self._onehot(row['fv_heavy_aho'])
        light = self._onehot(row['fv_light_aho'])
        target = self._onehot(row['ag'])
        target = F.pad(target, pad=(0, 0, 0, self.max_target_len - target.shape[0]), value=22)
        light = F.pad(light, pad=(0, 0, 0, 149 - light.shape[0]), value=22)
        x = torch.cat([heavy, light, target], dim=0).float()
        y = F.one_hot(torch.tensor(int(row['is_binder'])), num_classes=2).float()
        return x, y


class AbRosetta:
    """Antibody DomainBed (Tagasovska et al., 2024): four design rounds as environments."""
    N_STEPS = 15000
    CHECKPOINT_FREQ = 500
    num_classes = 2

    def __init__(self, root):
        candidates = [os.path.join(root or ".", name) for name in ("abdb.csv", "antibody_domainbed_dataset.csv")]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            raise FileNotFoundError(
                "Antibody DomainBed data not found. Looked for: " + ", ".join(candidates)
                + ". Download abdb.csv from https://zenodo.org/records/11446107 and pass its directory as --data_dir.")
        df = pd.read_csv(path)
        df.dropna(subset=['ddG'], inplace=True)
        df.drop_duplicates(subset=['fv_heavy_aho', 'fv_light_aho', 'fv_light_aho_seed', 'fv_heavy_aho_seed', 'ddG'],
                           inplace=True, keep='first')
        df.drop_duplicates(subset=['seq_id'], inplace=True, keep='last')
        df['is_binder'] = df['ddG'] < 0.0
        assert np.all(df['fv_heavy_aho'].apply(len).values == 149)
        assert np.all(np.isin(df['fv_light_aho'].apply(len).values, [148, 149]))
        self.max_target_len = df['ag'].apply(len).max()
        self.input_shape = (149 + 149 + self.max_target_len, 22)
        self.datasets = [SingleRound(df[df['env'] == env], self.max_target_len) for env in sorted(df['env'].unique())]

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)
