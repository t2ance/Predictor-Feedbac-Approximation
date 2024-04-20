import torch
from torch.utils.data import Dataset

from utils import padding_leading_zero


def sample_to_tensor(z_features, u_features, label, time_step_position=None):
    # TODO:
    # features = torch.cat((torch.tensor(idx).view(-1), z_features, u_features))
    features = torch.cat((z_features, u_features))

    return features, label


class SingleDataset(Dataset):
    def __init__(self, Z, U, D_steps):
        self.Z = Z
        self.U = U
        self.D_steps = D_steps

    def __len__(self):
        return len(self.Z) - self.D_steps

    def __getitem__(self, idx):
        z_features = self.Z[idx]
        u_features = padding_leading_zero(self.U, idx, self.D_steps).view(-1)
        label = self.Z[idx + self.D_steps]
        features, label = sample_to_tensor(z_features, u_features, label)
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
