import torch
from torch.utils.data import Dataset

from utils import padding_leading_zero


def sample_to_tensor(z_features, u_features, label, time_step_position):
    # TODO:
    # features = torch.cat((torch.tensor(idx).view(-1), z_features, u_features))
    features = torch.cat((torch.tensor(torch.nan).view(-1), z_features, u_features))

    return features, label


class ImplicitDataset(Dataset):
    def __init__(self, Z, U, D_steps):
        self.Z = Z
        self.U = U
        self.D_steps = D_steps

    def __len__(self):
        return len(self.U) - 2 * self.D_steps

    def __getitem__(self, idx):
        idx += self.D_steps
        z_features = self.Z[idx]
        u_features = self.U[idx - self.D_steps:idx].view(-1)
        label = self.Z[idx + self.D_steps]
        features, label = sample_to_tensor(z_features, u_features, label, -1)
        return features, label


class ExplictDataset(Dataset):
    def __init__(self, Z, U, P, D_steps):
        self.Z = Z
        self.U = U
        self.P = P
        self.D_steps = D_steps

    def __len__(self):
        return len(self.U) - self.D_steps

    def __getitem__(self, idx):
        idx += self.D_steps
        z_features = self.Z[idx]
        u_features = self.U[idx - self.D_steps:idx].view(-1)
        label = self.P[idx]
        features, label = sample_to_tensor(z_features, u_features, label, -1)
        return features, label


class PredictionDataset(Dataset):
    def __init__(self, all_samples):
        self.all_samples = all_samples

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        sample = self.all_samples[idx]
        return torch.tensor(sample[0], dtype=torch.float32), torch.tensor(sample[1], dtype=torch.float32)


if __name__ == '__main__':
    ...
