import torch
from neuralop.models import FNO1d


class PredictionFNO(torch.nn.Module):
    def __init__(self, n_modes_height: int, hidden_channels: int, in_features: int, out_features: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fno = FNO1d(n_modes_height=n_modes_height, hidden_channels=hidden_channels, in_channels=1, out_channels=1)
        self.projection = torch.nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-2)
        x = self.fno(x)
        x = self.projection(x)
        x = x.squeeze(-2)
        return x


if __name__ == '__main__':
    ...
