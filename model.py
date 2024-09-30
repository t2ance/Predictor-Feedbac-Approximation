import torch
from neuralop.models import FNO1d
from torch import nn


class LearningBasedPredictor(torch.nn.Module):
    def __init__(self, n_input: int, n_state: int, seq_len: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_input = n_input
        self.n_state = n_state
        self.seq_len = seq_len
        self.mse_loss = torch.nn.MSELoss()

    def compute(self, x: torch.Tensor):
        raise NotImplementedError()

    def forward(self, u: torch.Tensor, z: torch.Tensor, t: torch.Tensor, label: torch.Tensor = None, **kwargs):
        repeated_z = z.tile(u.shape[1]).reshape(z.shape[0], -1, z.shape[1])
        x = torch.concatenate([u, repeated_z], -1)
        outs = self.compute(x)
        out = outs[:, -1, :]
        if label is not None:
            return out, self.mse_loss(outs, label)
        else:
            return out


class FFN(LearningBasedPredictor):
    def __init__(self, n_layers: int, layer_width: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_features = self.n_state + self.seq_len * self.n_input + 1
        out_features = self.n_state
        layers = [
            torch.nn.Linear(in_features=in_features, out_features=layer_width * in_features),
            torch.nn.ReLU()
        ]
        for _ in range(n_layers):
            layers.extend([
                torch.nn.Linear(in_features=layer_width * in_features, out_features=layer_width * in_features),
                torch.nn.ReLU()
            ])
        layers.extend([
            torch.nn.Linear(in_features=layer_width * in_features, out_features=in_features),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=in_features, out_features=out_features)
        ])

        self.projection = torch.nn.Sequential(*layers)

    def compute(self, x: torch.Tensor):
        return self.projection(x)


class GRUNet(LearningBasedPredictor):
    def __init__(self, hidden_size, num_layers, output_size, **kwargs):
        super(GRUNet, self).__init__(**kwargs)
        self.rnn = nn.GRU(self.n_input + self.n_state, hidden_size, num_layers, batch_first=True)
        self.projection = nn.Linear(hidden_size, output_size)

    def compute(self, x: torch.Tensor):
        output, (_) = self.rnn(x)
        x = self.projection(output)
        return x


class LSTMNet(LearningBasedPredictor):
    def __init__(self, hidden_size, num_layers, output_size, **kwargs):
        super(LSTMNet, self).__init__(**kwargs)
        self.rnn = nn.LSTM(self.n_input + self.n_state, hidden_size, num_layers, batch_first=True)
        self.projection = nn.Linear(hidden_size, output_size)

    def compute(self, x: torch.Tensor):
        output, (_) = self.rnn(x)
        x = self.projection(output)
        return x


def fft_transform(input_tensor, target_length):
    fft_result = torch.fft.rfft(input_tensor)
    current_length = fft_result.shape[1]

    if current_length > target_length:
        output = fft_result[:, :target_length]
    elif current_length < target_length:
        padding = torch.zeros((input_tensor.shape[0], target_length - current_length), device=input_tensor.device)
        output = torch.cat((fft_result, padding), dim=1)
    else:
        output = fft_result

    output = torch.concatenate([output.real, output.imag], -1)
    return output.to(input_tensor.device)


def compl_mul(input_tensor, weights):
    """
    Complex multiplication:
    (batch, in_channel, ...), (in_channel, out_channel, ...) -> (batch, out_channel, ...), where ``...'' represents the spatial part of the input.
    """
    return torch.einsum("bi...,io...->bo...", input_tensor, weights)


def resize_rfft(ar, s):
    """
    Truncates or zero pads the highest frequencies of ``ar'' such that torch.fft.irfft(ar, n=s) is either an interpolation to a finer grid or a subsampling to a coarser grid.
    Args
        ar: (..., N) tensor, must satisfy real conjugate symmetry (not checked)
        s: (int), desired irfft output dimension >= 1
    Output
        out: (..., s//2 + 1) tensor
    """
    N = ar.shape[-1]
    s = s // 2 + 1 if s >= 1 else s // 2
    if s >= N:  # zero pad or leave alone
        out = torch.zeros(list(ar.shape[:-1]) + [s - N], dtype=torch.cfloat, device=ar.device)
        out = torch.cat((ar[..., :N], out), dim=-1)
    elif s >= 1:  # truncate
        out = ar[..., :s]
    else:  # edge case
        raise ValueError("s must be greater than or equal to 1.")

    return out


class TimeAwareNeuralOperator(LearningBasedPredictor):
    def __init__(self, ffn: str, rnn: str, invert: bool, params, **kwargs):
        super().__init__(**kwargs)
        self.invert = invert
        if invert:
            if rnn == 'GRU':
                self.rnn = nn.GRU(self.n_state + self.n_input, params['gru_hidden_size'], params['gru_n_layers'],
                                  batch_first=True)
                hidden_size = params['gru_hidden_size']
            elif rnn == 'LSTM':
                self.rnn = nn.LSTM(self.n_state + self.n_input, params['lstm_hidden_size'], params['lstm_n_layers'],
                                   batch_first=True)
                hidden_size = params['lstm_hidden_size']
            else:
                raise NotImplementedError()

            if ffn == 'FNO':
                self.ffn = FNOProjection(n_modes_height=params['n_modes_height'], n_layers=params['fno_n_layers'],
                                         hidden_channels=params['hidden_channels'], n_input_channel=hidden_size,
                                         n_out_channel=self.n_state, **kwargs)
            elif ffn == 'DeepONet':
                self.ffn = DeepONet(hidden_size=params['deeponet_hidden_size'], n_layer=params['deeponet_n_layer'],
                                    n_input_channel=hidden_size, n_output_channel=self.n_state, **kwargs)
            else:
                raise NotImplementedError()
        else:
            if ffn == 'FNO':
                self.ffn = FNOProjection(n_modes_height=params['n_modes_height'], n_layers=params['fno_n_layers'],
                                         hidden_channels=params['hidden_channels'], **kwargs)
            elif ffn == 'DeepONet':
                self.ffn = DeepONet(hidden_size=params['deeponet_hidden_size'], n_layer=params['deeponet_n_layer'],
                                    **kwargs)
            else:
                raise NotImplementedError()

            if rnn == 'GRU':
                self.rnn = nn.GRU(self.n_state, params['gru_hidden_size'], params['gru_n_layers'], batch_first=True)
                self.projection = nn.Linear(params['gru_hidden_size'], self.n_state)
            elif rnn == 'LSTM':
                self.rnn = nn.LSTM(self.n_state, params['lstm_hidden_size'], params['lstm_n_layers'], batch_first=True)
                self.projection = nn.Linear(params['lstm_hidden_size'], self.n_state)
            else:
                raise NotImplementedError()

    def compute(self, x: torch.Tensor):
        if self.invert:
            rnn_out, _ = self.rnn(x)
            ffn_out = self.ffn.compute(rnn_out)
            return ffn_out
        else:
            ffn_out = self.ffn.compute(x)
            output, _ = self.rnn(ffn_out)
            rnn_out = self.projection(output)
            return rnn_out


class FNOProjection(LearningBasedPredictor):
    def __init__(self, n_modes_height: int, hidden_channels: int, n_layers: int, n_input_channel=None,
                 n_out_channel=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_modes_height = n_modes_height
        if n_input_channel is None:
            n_input_channel = self.n_input + self.n_state
        if n_out_channel is None:
            n_out_channel = self.n_state
        self.fno = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                         in_channels=n_input_channel, out_channels=n_out_channel)

    def compute(self, x: torch.Tensor):
        x = self.fno(x.transpose(1, 2))
        return x.transpose(1, 2)


class DeepONet(LearningBasedPredictor):
    def __init__(self, hidden_size, n_layer, n_input_channel=None, n_output_channel=None, *args, **kwargs):
        super(DeepONet, self).__init__(*args, **kwargs)
        if n_input_channel is None:
            n_input_channel = self.n_state + self.n_input
        if n_output_channel is None:
            n_output_channel = self.n_state
        self.n_layer = n_layer
        self.hidden_size = hidden_size
        self.branch_net = BranchNet(n_input_channel=n_input_channel, seq_len=self.seq_len,
                                    hidden_size=hidden_size, n_layer=n_layer)
        self.trunk_net = TrunkNet(y_dim=1, hidden_size=hidden_size, n_layer=n_layer)
        self.output_layer = nn.Linear(self.seq_len * self.hidden_size, n_output_channel)

    def compute(self, x: torch.Tensor):
        """
        x: (batch_size, n_input_channel, seq_len)
        y: (seq_len, y_dim)
        """
        y = torch.linspace(0, 1, steps=self.seq_len, device=x.device).unsqueeze(1)
        branch_out = self.branch_net(x)  # (batch_size, branch_size)
        trunk_out = self.trunk_net(y)  # (seq_len, trunk_size)

        # branch_size == trunk_size == hidden_size
        branch_out = branch_out.unsqueeze(1)  # (batch_size, 1, branch_size)
        trunk_out = trunk_out.unsqueeze(0)  # (1, seq_len, trunk_size)

        combined = branch_out * trunk_out  # (batch_size, seq_len, branch_size)

        combined = combined.view(combined.size(0), -1)  # (batch_size, seq_len * branch_size)

        output = self.output_layer(combined)  # (batch_size, n_output_channel)

        output = output.unsqueeze(2).repeat(1, 1, y.size(0))  # (batch_size, n_output_channel, seq_len)

        return output.transpose(1, 2)


class BranchNet(nn.Module):
    def __init__(self, n_input_channel, seq_len, hidden_size, n_layer):
        super(BranchNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(n_input_channel * seq_len, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        # x shape: (batch_size, n_input_channel, seq_len)
        x = self.flatten(x)  # (batch_size, n_input_channel * seq_len)
        out = self.fc(x)  # (batch_size, hidden_size)
        return out


class TrunkNet(nn.Module):
    def __init__(self, y_dim, hidden_size, n_layer):
        super(TrunkNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(y_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, y):
        # y shape: (seq_len, y_dim)
        out = self.fc(y)  # (seq_len, hidden_size)
        return out


if __name__ == '__main__':
    ...
