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
                self.rnn = nn.GRU(self.n_state+self.n_input, params['gru_hidden_size'], params['gru_n_layers'], batch_first=True)
                hidden_size = params['gru_hidden_size']
                # self.projection = nn.Linear(params['gru_hidden_size'], self.n_state)
            elif rnn == 'LSTM':
                self.rnn = nn.LSTM(self.n_state+self.n_input, params['lstm_hidden_size'], params['lstm_n_layers'], batch_first=True)
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


class LinearDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        """
        Fourier neural decoder layer for functions over the torus. Maps vectors to functions.
        Inputs:
            in_channels  (int): dimension of input vectors
            out_channels (int): total number of functions to extract
        """
        super(LinearDecoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes

        self.scale = 1. / (self.in_channels * self.out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.modes + 1, dtype=torch.cfloat))

    def forward(self, x, s):
        """
        Input shape (of x):     (batch, in_channels, ...)
        s            (int):     desired spatial resolution (nx,) of functions
        Output shape:           (batch, out_channels, ..., nx)
        """
        # Multiply relevant Fourier modes
        x = compl_mul(x[..., None].type(torch.cfloat), self.weights)

        # Zero pad modes
        x = resize_rfft(x, s)

        # Return to physical space
        return torch.fft.irfft(x, n=s)


class LinearFunctional(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        """
        Fourier neural functionals (encoder) layer for functions over the torus. Maps functions to vectors.
        Inputs:
            in_channels  (int): number of input functions
            out_channels (int): total number of linear functionals to extract
        """
        super(LinearFunctional, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes

        # Complex conjugation in L^2 inner product is absorbed into parametrization
        self.scale = 1. / (self.in_channels * self.out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(self.in_channels, self.out_channels, self.modes + 1, dtype=torch.cfloat))

    def forward(self, x):
        """
        Input shape (of x):     (batch, in_channels, ..., nx_in)
        Output shape:           (batch, out_channels, ...)
        """
        # Compute Fourier coeffcients (scaled to approximate integration)
        x = torch.fft.rfft(x, norm="forward")

        # Truncate input modes
        x = resize_rfft(x, 2 * self.modes)

        # Multiply relevant Fourier modes and take the real part
        x = compl_mul(x, self.weights).real

        # Integrate the conjugate product in physical space by summing Fourier coefficients
        return 2 * torch.sum(x, dim=-1) - x[..., 0]


class DeepONet(LearningBasedPredictor):

    def __init__(self, hidden_size: int, n_layer: int, **kwargs):
        super(DeepONet, self).__init__(**kwargs)
        # creating the branch network#
        self.branch_net = MLP(n_input=self.seq_len, hidden_size=hidden_size, depth=n_layer)
        self.branch_net.float()

        # creating the trunk network#
        self.trunk_nets = nn.ParameterList()
        self.biases = nn.ParameterList()
        for _ in range(self.n_state):
            trunk_net = MLP(n_input=1, hidden_size=hidden_size, depth=n_layer).float()
            bias = nn.Parameter(torch.ones((1,)), requires_grad=True)

            self.trunk_nets.append(trunk_net)
            self.biases.append(bias)

    def compute(self, u: torch.Tensor, z: torch.Tensor, t: torch.Tensor):
        repeated_z = z.tile(u.shape[1]).reshape(z.shape[0], -1, z.shape[1])
        x = torch.concatenate([u, repeated_z], -1)
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1).reshape(-1, self.seq_len)
        new_batch_size = x.shape[0]

        x_trunk = torch.arange(self.seq_len, device=x.device, dtype=torch.float32).tile(new_batch_size).reshape(
            new_batch_size, -1, 1)

        x_i = self.forward_(x, x_trunk)

        return x_i.reshape(batch_size, self.seq_len, -1)

    def forward_(self, x_branch, x_trunk):
        branch_out = self.branch_net(x_branch)

        # trunk网络的输出（批量计算）
        trunk_outputs = []
        for trunk_net in self.trunk_nets:
            trunk_output = trunk_net(x_trunk)
            trunk_outputs.append(trunk_output)

        trunk_outputs = torch.stack(trunk_outputs, dim=-1)  # 形状: [batch_size, seq_len, n_state]

        # 广播branch_out，以便与trunk_outputs进行批量计算
        branch_out_expanded = branch_out.unsqueeze(1).repeat(1, self.seq_len, 1)

        # 批量点积计算
        output = torch.einsum('bsi,bsi->bs', branch_out_expanded, trunk_outputs)

        # 添加biases
        biases = torch.stack(self.biases).unsqueeze(0).unsqueeze(0)  # 形状: [1, 1, n_state]
        output += biases  # 形状: [batch_size, seq_len, n_state]

        return output

    # def compute(self, u: torch.Tensor, z: torch.Tensor, t: torch.Tensor):
    #     batch_size = u.shape[0]
    #     x = torch.hstack([u.reshape(batch_size, -1), z.reshape(batch_size, -1)])
    #     out = []
    #     for i in range(self.seq_len):
    #         x_i = self.forward_(x, i * torch.ones(x.shape[0], 1, device=x.device))
    #         out.append(x_i.unsqueeze(1))
    #     return torch.concatenate(out, dim=1)
    #
    # def forward_(self, x_branch, x_trunk):
    #     branch_out = self.branch_net(x_branch)
    #     outputs = []
    #     for trunk_net, bias in zip(self.trunk_nets, self.biases):
    #         trunk_out = trunk_net(x_trunk, final_act=True)
    #         output = torch.einsum('ij,ij->i', branch_out, trunk_out) + bias
    #         outputs.append(output)
    #     return torch.vstack(outputs).T


class MLP(nn.Module):
    def __init__(self, n_input, hidden_size, depth, act=None):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        # the activation function#
        if act is None:
            act = nn.ReLU()
        self.act = act

        # Input layer
        self.layers.append(nn.Linear(n_input, hidden_size))

        for _ in range(depth - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

    def forward(self, x, final_act=False):
        for i in range(len(self.layers) - 1):
            x = self.act(self.layers[i](x))
        x = self.layers[-1](x)  # No activation after the last layer

        if not final_act:
            return x
        else:
            return torch.relu(x)


if __name__ == '__main__':
    ...
