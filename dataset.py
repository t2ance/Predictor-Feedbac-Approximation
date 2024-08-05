import torch
from torch.utils.data import Dataset


def sample_to_tensor(z_features, u_features, time_step_position):
    if z_features is not torch.Tensor:
        z_features = torch.tensor(z_features)
    if u_features is not torch.Tensor:
        u_features = torch.tensor(u_features)
    features = torch.cat((torch.tensor(time_step_position).view(-1), z_features, u_features.view(-1)))
    return features


class ZUZDataset(Dataset):
    def __init__(self, Z, U, n_point_delay, dt):
        self.Z = Z
        self.U = U
        self.n_point_delay = n_point_delay
        self.dt = dt
        raise NotImplementedError()

    def __len__(self):
        return len(self.U) - 2 * self.n_point_delay

    def __getitem__(self, idx):
        idx += self.n_point_delay
        z_features = self.Z[idx]
        u_features = self.U[idx - self.n_point_delay:idx].view(-1)
        label = self.Z[idx + self.n_point_delay]
        features = sample_to_tensor(z_features, u_features, idx * self.dt)
        return features, label


class ZUPDataset(Dataset):
    def __init__(self, Z, U, P, n_point_delay, dt, ts):
        self.Z = Z
        self.U = U
        self.P = P
        self.dt = dt
        self.ts = ts
        self.n_point_delay = n_point_delay
        self.n_point_start = n_point_delay(0)

    def __len__(self):
        return len(self.U) - self.n_point_start

    def __getitem__(self, idx):
        idx += self.n_point_start
        t = self.ts[idx]
        z_features = self.Z[idx]
        u_features = self.U[idx - self.n_point_delay(t):idx]
        label = self.P[idx]
        features = sample_to_tensor(z_features, u_features, t)
        return features, label


class PredictionDataset(Dataset):
    def __init__(self, all_samples):
        self.all_samples = all_samples

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        sample = self.all_samples[idx]
        return sample[0].to(dtype=torch.float32), sample[1].to(dtype=torch.float32)


if __name__ == '__main__':
    ...
