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

    def compute(self, u: torch.Tensor, z: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError()

    def forward(self, u: torch.Tensor, z: torch.Tensor, t: torch.Tensor, label: torch.Tensor = None, **kwargs):
        if isinstance(self, TimeAwareNeuralOperator):
            # x_ffn, x_rnn = self.compute(u, z, t, ffn_output=True)
            # x_rnn = x_rnn + x_ffn
            # out = x_rnn[:, -1, :]
            # if label is not None:
            #     return out, self.mse_loss(x_ffn, label) + self.mse_loss(x_rnn, label)
            # else:
            #     return out
            outs = self.compute(u, z, t)
            out = outs[:, -1, :]
            if label is not None:
                return out, self.mse_loss(outs, label)
            else:
                return out
        else:
            x = self.compute(u, z, t)
            out = x[:, -1, :]
            if label is not None:
                return out, self.mse_loss(x, label)
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

    def compute(self, u: torch.Tensor, z: torch.Tensor, t: torch.Tensor):
        return self.projection()


class GRUNet(LearningBasedPredictor):
    def __init__(self, hidden_size, num_layers, output_size, **kwargs):
        super(GRUNet, self).__init__(**kwargs)
        self.rnn = nn.GRU(self.n_input + self.n_state, hidden_size, num_layers, batch_first=True)
        self.projection = nn.Linear(hidden_size, output_size)

    def compute(self, u: torch.Tensor, z: torch.Tensor, t: torch.Tensor):
        repeated_z = z.tile(u.shape[1]).reshape(z.shape[0], -1, z.shape[1])
        x = torch.concatenate([u, repeated_z], -1)
        output, (_) = self.rnn(x)
        x = self.projection(output)
        return x


class LSTMNet(LearningBasedPredictor):
    def __init__(self, hidden_size, num_layers, output_size, **kwargs):
        super(LSTMNet, self).__init__(**kwargs)
        self.rnn = nn.LSTM(self.n_input + self.n_state, hidden_size, num_layers, batch_first=True)
        self.projection = nn.Linear(hidden_size, output_size)

    def compute(self, u: torch.Tensor, z: torch.Tensor, t: torch.Tensor):
        repeated_z = z.tile(u.shape[1]).reshape(z.shape[0], -1, z.shape[1])
        x = torch.concatenate([u, repeated_z], -1)
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
                # self.projection = nn.Linear(params['gru_hidden_size'], self.n_state)
            elif rnn == 'LSTM':
                self.rnn = nn.LSTM(self.n_state + self.n_input, params['lstm_hidden_size'], params['lstm_n_layers'],
                                   batch_first=True)
                hidden_size = params['lstm_hidden_size']
                # self.projection = nn.Linear(params['lstm_hidden_size'], self.n_state)
            else:
                raise NotImplementedError()

            if ffn == 'FNO':
                self.ffn = FNO1d(n_modes_height=params['n_modes_height'], n_layers=params['fno_n_layers'],
                                 hidden_channels=params['hidden_channels'], in_channels=hidden_size,
                                 out_channels=self.n_state)
            elif ffn == 'DeepONet':
                ...
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

    def compute(self, u: torch.Tensor, z: torch.Tensor, t: torch.Tensor, ffn_output: bool = False):
        if self.invert:
            repeated_z = z.tile(u.shape[1]).reshape(z.shape[0], -1, z.shape[1])
            x = torch.concatenate([u, repeated_z], -1)
            rnn_out, _ = self.rnn(x)
            ffn_out = self.ffn(rnn_out.transpose(1, 2)).transpose(1, 2)
            # if ffn_output:
            #     return ffn_out, rnn_out
            return ffn_out
        else:
            ffn_out = self.ffn.compute(u, z, t)
            output, _ = self.rnn(ffn_out)
            rnn_out = self.projection(output)
            if ffn_output:
                return ffn_out, rnn_out
            return rnn_out


class FNOProjection(LearningBasedPredictor):
    def __init__(self, n_modes_height: int, hidden_channels: int, n_layers: int, in_channels=None, out_channels=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_modes_height = n_modes_height
        if in_channels is None:
            in_channels = self.n_input + self.n_state
        # in_channels += 1
        if out_channels is None:
            out_channels = self.n_state
        self.fno = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                         in_channels=in_channels, out_channels=out_channels)
        # self.linear_decoder = LinearDecoder(self.n_state, 1, n_modes_height)

    def compute(self, u: torch.Tensor, z: torch.Tensor, t: torch.Tensor):
        repeated_z = z.tile(u.shape[1]).reshape(z.shape[0], -1, z.shape[1])
        x = torch.concatenate([u, repeated_z], -1)
        # u = u.transpose(1, 2)
        # seq_len = u.shape[-1]
        # z = self.linear_decoder(z, seq_len)
        # x = torch.concatenate([u, z], 1)
        x = self.fno(x.transpose(1, 2))
        return x.transpose(1, 2)


class BranchNet(nn.Module):
    def __init__(self, n_input_channel, seq_len, hidden_size):
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
    def __init__(self, y_dim, hidden_size):
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


class DeepONet(nn.Module):
    def __init__(self, branch_net, trunk_net, hidden_size, n_output_channel):
        super(DeepONet, self).__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.output_layer = nn.Linear(hidden_size * hidden_size, n_output_channel)

    def forward(self, x, y):
        """
        x: (batch_size, n_input_channel, seq_len)
        y: (seq_len, y_dim)
        """
        branch_out = self.branch_net(x)  # (batch_size, branch_size)
        trunk_out = self.trunk_net(y)  # (seq_len, trunk_size)

        # 假设 branch_size == trunk_size == hidden_size
        # 扩展维度以进行广播
        branch_out = branch_out.unsqueeze(1)  # (batch_size, 1, branch_size)
        trunk_out = trunk_out.unsqueeze(0)  # (1, seq_len, trunk_size)

        # 结合分支和干网络嵌入
        combined = branch_out * trunk_out  # (batch_size, seq_len, branch_size)

        # 展平最后两个维度以输入到输出层
        combined = combined.view(combined.size(0), -1)  # (batch_size, seq_len * branch_size)

        output = self.output_layer(combined)  # (batch_size, n_output_channel)

        # 将输出扩展到每个位置
        output = output.unsqueeze(2).repeat(1, 1, y.size(0))  # (batch_size, n_output_channel, seq_len)

        return output


if __name__ == '__main__':
    ...
