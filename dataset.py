import torch
from torch.utils.data import Dataset


class SingleDataset(Dataset):
    def __init__(self, Z, U, D_steps):
        self.Z = Z
        self.U = U
        self.D_steps = D_steps

    def __len__(self):
        return len(self.Z) - 2 * self.D_steps

    def __getitem__(self, idx):
        z_features = self.Z[idx + self.D_steps]
        u_features = self.U[idx:idx + self.D_steps].view(-1)
        features = torch.cat((torch.tensor(idx).view(-1), z_features, u_features))

        label = self.Z[idx + 2 * self.D_steps]
        return features, label


class PredictionDataset(Dataset):
    def __init__(self, all_samples):
        self.all_samples = all_samples

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        return self.all_samples[idx]


if __name__ == '__main__':
    ...
