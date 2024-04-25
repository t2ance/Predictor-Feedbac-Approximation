import torch
from torch.utils.data import Dataset


def sample_to_tensor(z_features, u_features, time_step_position):
    features = torch.cat((torch.tensor(time_step_position).view(-1), z_features, u_features))
    return features


class ImplicitDataset(Dataset):
    def __init__(self, Z, U, D_steps, dt):
        self.Z = Z
        self.U = U
        self.D_steps = D_steps
        self.dt = dt

    def __len__(self):
        return len(self.U) - 2 * self.D_steps

    def __getitem__(self, idx):
        idx += self.D_steps
        z_features = self.Z[idx]
        u_features = self.U[idx - self.D_steps:idx].view(-1)
        label = self.Z[idx + self.D_steps]
        features = sample_to_tensor(z_features, u_features, idx * self.dt)
        return features, label


class ExplictDataset(Dataset):
    def __init__(self, Z, U, P, n_delay_point, dt):
        self.Z = Z
        self.U = U
        self.P = P
        self.dt = dt
        self.n_delay_point = n_delay_point

    def __len__(self):
        return len(self.U) - self.n_delay_point

    def __getitem__(self, idx):
        idx += self.n_delay_point
        z_features = self.Z[idx]
        u_features = self.U[idx - self.n_delay_point:idx].view(-1)
        label = self.P[idx]
        features = sample_to_tensor(z_features, u_features, idx * self.dt)
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
