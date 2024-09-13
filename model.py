import torch
from neuralop.models import FNO1d
from torch import nn


class FFN(torch.nn.Module):
    def __init__(self, n_state: int, n_point_delay: int, n_input: int, n_layers: int, layer_width: int, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.mse_loss = torch.nn.MSELoss()
        in_features = n_state + n_point_delay * n_input + 1
        out_features = n_state
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

    def forward(self, x: torch.Tensor, label: torch.Tensor = None):
        x = self.projection(x)
        if label is None:
            return x
        return x, self.mse_loss(x, label)


class GRUNet(nn.Module):
    def __init__(self, hidden_size, num_layers, output_size):
        super(GRUNet, self).__init__()
        self.rnn = nn.GRU(1, hidden_size, num_layers, batch_first=True)
        self.projection = nn.Linear(hidden_size, output_size)
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        if x.ndim == 2:
            x = x[:, :, None]
        output, (_) = self.rnn(x)
        x = self.projection(output[:, -1, :])

        if labels is None:
            return x
        return x, self.mse_loss(x, labels)


class LSTMNet(nn.Module):
    def __init__(self, hidden_size, num_layers, output_size):
        super(LSTMNet, self).__init__()
        self.rnn = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        self.projection = nn.Linear(hidden_size, output_size)
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        if x.ndim == 2:
            x = x[:, :, None]
        output, (_, _) = self.rnn(x)
        x = self.projection(output[:, -1, :])

        if labels is None:
            return x
        return x, self.mse_loss(x, labels)


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


class TimeAwareNeuralOperator(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor, label: torch.Tensor = None):
        x = self.ffn(x)
        output, hidden = self.rnn(x)
        x = self.projection(output[:, -1, :])

        if label is None:
            return x
        return x, self.mse_loss(x, label)


class FNOGRU(TimeAwareNeuralOperator):

    def __init__(self, n_modes_height: int, hidden_channels: int, fno_n_layers: int, gru_n_layers: int,
                 gru_hidden_size: int, n_state: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ffn = FNOProjection(n_modes_height=n_modes_height, hidden_channels=hidden_channels, n_state=n_state,
                                 n_layers=fno_n_layers, function_out=True)
        self.rnn = nn.GRU(1, gru_hidden_size, gru_n_layers, batch_first=True)
        self.projection = nn.Linear(gru_hidden_size, n_state)


class FNOLSTM(TimeAwareNeuralOperator):

    def __init__(self, n_modes_height: int, hidden_channels: int, fno_n_layers: int, lstm_n_layers: int,
                 lstm_hidden_size: int, n_state: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ffn = FNOProjection(n_modes_height=n_modes_height, hidden_channels=hidden_channels, n_state=n_state,
                                 n_layers=fno_n_layers, function_out=True)
        self.rnn = nn.LSTM(1, lstm_hidden_size, lstm_n_layers, batch_first=True)
        self.projection = nn.Linear(lstm_hidden_size, n_state)


class DeepONetGRU(TimeAwareNeuralOperator):
    def __init__(self, n_point_start: int, n_input: int, deeponet_n_layer: int, deeponet_hidden_size: int,
                 gru_n_layers: int, gru_hidden_size: int, n_state: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ffn = DeepONet(n_input_branch=n_point_start * n_input + n_state, n_input_trunk=1, n_point=n_point_start,
                            layer_width=deeponet_hidden_size, n_layer=deeponet_n_layer, n_output=1, function_out=True)
        self.rnn = nn.GRU(1, gru_hidden_size, gru_n_layers, batch_first=True)
        self.projection = nn.Linear(gru_hidden_size, n_state)


class DeepONetLSTM(TimeAwareNeuralOperator):
    def __init__(self, n_point_start: int, n_input: int, deeponet_n_layer: int, deeponet_hidden_size: int,
                 lstm_n_layers: int, lstm_hidden_size: int, n_state: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ffn = DeepONet(n_input_branch=n_point_start * n_input + n_state, n_input_trunk=1, n_point=n_point_start,
                            layer_width=deeponet_hidden_size, n_layer=deeponet_n_layer, n_output=1, function_out=True)
        self.rnn = nn.LSTM(1, lstm_hidden_size, lstm_n_layers, batch_first=True)
        self.projection = nn.Linear(lstm_hidden_size, n_state)


################################
# Old models, not used anymore #
################################

#
# class GRUNet(nn.Module):
#     def __init__(self, input_size, layer_width, num_layers, output_size):
#         super(GRUNet, self).__init__()
#         self.hidden_size = layer_width * input_size
#         self.num_layers = num_layers
#
#         self.gru = nn.GRU(input_size, self.hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(self.hidden_size, output_size)
#         self.mse_loss = torch.nn.MSELoss()
#         self.hidden = None
#
#     def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
#         if self.hidden is None:
#             self.hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         if x.ndim == 2:
#             x = x[:, None, :]
#
#         out, hidden = self.gru(x, self.hidden)
#         self.hidden = hidden.detach()
#
#         out = self.fc(out[:, -1, :])
#         if labels is None:
#             return out
#         return out, self.mse_loss(out, labels)
#
#     def reset_state(self):
#         self.hidden = None
#
#
# class LSTMNet(nn.Module):
#     def __init__(self, input_size, layer_width, num_layers, output_size):
#         super(LSTMNet, self).__init__()
#         self.hidden_size = layer_width * input_size
#         self.num_layers = num_layers
#
#         self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(self.hidden_size, output_size)
#         self.mse_loss = torch.nn.MSELoss()
#         self.hidden = None
#         self.cell = None
#
#     def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
#         if self.hidden is None or self.cell is None:
#             self.hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#             self.cell = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         if x.ndim == 2:
#             x = x[:, None, :]
#         out, (hidden, cell) = self.lstm(x, (self.hidden, self.cell))
#         self.hidden = hidden.detach()
#         self.cell = cell.detach()
#
#         out = self.fc(out[:, -1, :])
#         if labels is None:
#             return out
#         return out, self.mse_loss(out, labels)
#
#     def reset_state(self):
#         self.hidden = None
#         self.cell = None
#

# class TimeAwareFFN(torch.nn.Module):
#     def __init__(self, residual, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.mse_loss = torch.nn.MSELoss()
#         self.residual = residual
#
#     def forward(self, x: torch.Tensor, label: torch.Tensor = None, ffn_out: bool = False):
#
#         ffn_x = self.ffn(x)
#         if self.residual:
#             rnn_x = self.rnn(ffn_x) + ffn_x
#         else:
#             rnn_x = self.rnn(ffn_x)
#         if ffn_out:
#             if label is None:
#                 return rnn_x, ffn_x
#             return rnn_x, ffn_x, self.mse_loss(rnn_x, label), self.mse_loss(ffn_x, label)
#         else:
#             if label is None:
#                 return rnn_x
#             return rnn_x, self.mse_loss(rnn_x, label)
#
#     def reset_state(self):
#         self.rnn.reset_state()
#
#
# class FNOProjectionGRU(TimeAwareFFN):
#     def __init__(self, n_modes_height: int, hidden_channels: int, fno_n_layers: int, gru_n_layers: int,
#                  gru_layer_width: int, n_state: int, ffn=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         if ffn is None:
#             ffn = FNOProjection(n_modes_height=n_modes_height, hidden_channels=hidden_channels, n_state=n_state,
#                                 n_layers=fno_n_layers)
#         else:
#             print('Pretrained FNO model loaded to FNO-GRU')
#         self.ffn = ffn
#         self.rnn = GRUNet(input_size=n_state, layer_width=gru_layer_width, num_layers=gru_n_layers,
#                           output_size=n_state)
#
#
# class FNOProjectionLSTM(TimeAwareFFN):
#     def __init__(self, n_modes_height: int, hidden_channels: int, fno_n_layers: int, lstm_n_layers: int,
#                  lstm_layer_width: int, n_state: int, ffn=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         if ffn is None:
#             ffn = FNOProjection(n_modes_height=n_modes_height, hidden_channels=hidden_channels, n_state=n_state,
#                                 n_layers=fno_n_layers)
#         else:
#             print('Pretrained FNO model loaded to FNO-LSTM')
#         self.ffn = ffn
#         self.rnn = LSTMNet(input_size=n_state, layer_width=lstm_layer_width, num_layers=lstm_n_layers,
#                            output_size=n_state)


# class DeepONetGRU(TimeAwareFFN):
#     def __init__(self, n_point_start: int, n_input: int, deeponet_n_layer: int, deeponet_hidden_size: int,
#                  gru_n_layers: int, gru_layer_width: int, n_state: int, ffn=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         if ffn is None:
#             ffn = DeepONet(n_input_branch=n_point_start * n_input, n_input_trunk=n_state,
#                            layer_width=deeponet_hidden_size, n_layer=deeponet_n_layer,
#                            n_output=n_state)
#         else:
#             print('Pretrained DeepONet model loaded to DeepONet-GRU')
#         self.ffn = ffn
#         self.rnn = GRUNet(input_size=n_state, layer_width=gru_layer_width, num_layers=gru_n_layers,
#                           output_size=n_state)
#
#
# class DeepONetLSTM(TimeAwareFFN):
#     def __init__(self, n_point_start: int, n_input: int, deeponet_n_layer: int, deeponet_hidden_size: int,
#                  lstm_n_layers: int, lstm_layer_width: int, n_state: int, ffn=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         if ffn is None:
#             ffn = DeepONet(n_input_branch=n_point_start * n_input, n_input_trunk=n_state,
#                            layer_width=deeponet_hidden_size, n_layer=deeponet_n_layer,
#                            n_output=n_state)
#         else:
#             print('Pretrained DeepONet model loaded to DeepONet-GRU')
#         self.ffn = ffn
#         self.rnn = LSTMNet(input_size=n_state, layer_width=lstm_layer_width, num_layers=lstm_n_layers,
#                            output_size=n_state)


class FNOProjection(torch.nn.Module):
    def __init__(self, n_modes_height: int, hidden_channels: int, n_layers: int, n_state: int,
                 function_out: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.function_out = function_out
        self.n_state = n_state
        self.mse_loss = torch.nn.MSELoss()
        self.n_modes_height = n_modes_height
        self.fno = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                         in_channels=2, out_channels=1)
        self.linear_decoder = LinearDecoder(n_state, 1, n_modes_height)
        if not self.function_out:
            self.linear_functional = LinearFunctional(1, n_state, n_modes_height)

    def forward(self, x: torch.Tensor, label: torch.Tensor = None):
        u = x[:, self.n_state:].unsqueeze(-2)
        z = x[:, :self.n_state]
        z = self.linear_decoder(z, u.shape[-1])
        x = torch.concatenate([u, z], 1)
        x = self.fno(x)
        if self.function_out:
            return x.transpose(1, 2)
        x = self.linear_functional(x)

        if label is None:
            return x
        return x, self.mse_loss(x, label)


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


class DeepONet(nn.Module):
    """
        Implementation of the Deep Operator Network
    """

    def __init__(self, n_input_branch: int, n_input_trunk: int, layer_width: int, n_layer: int, n_output: int,
                 function_out: bool = False, n_point=None):
        """
            Creates the DON using the following parameters

            Parameters:
            n_branch (int) : the input size of the branch network
            n_trunk  (int) : the input size of the trunk network
            depth    (int) : number of layers in each network
            width.   (int) : number of nodes at each layer
            n_output (int) : output dimension of network
        """
        super(DeepONet, self).__init__()
        self.function_out = function_out
        self.n_point = n_point
        self.n_output = n_output
        # creating the branch network#
        self.branch_net = MLP(n_input=n_input_branch, hidden_size=layer_width, depth=n_layer)
        self.branch_net.float()

        # creating the trunk network#
        self.trunk_nets = nn.ParameterList()
        self.biases = nn.ParameterList()
        for _ in range(n_output):
            trunk_net = MLP(n_input=n_input_trunk, hidden_size=layer_width, depth=n_layer).float()
            bias = nn.Parameter(torch.ones((1,)), requires_grad=True)

            self.trunk_nets.append(trunk_net)
            self.biases.append(bias)

        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor, label: torch.Tensor = None):
        if self.function_out:
            out = []
            for i in range(self.n_point):
                x_i = self.forward_(x, i * torch.ones(x.shape[0], 1, device=x.device))
                out.append(x_i.unsqueeze(1))
            return torch.concatenate(out, dim=1)
        else:
            x = self.forward_(x[:, self.n_output:], x[:, :self.n_output])
            if label is None:
                return x
            return x, self.mse_loss(x, label)

    def forward_(self, x_branch, x_trunk):
        """
            evaluates the operator

            x_branch : input_function
            x_trunk : point evaluating at

            returns a scalar
        """

        branch_out = self.branch_net(x_branch)
        outputs = []
        for trunk_net, bias in zip(self.trunk_nets, self.biases):
            trunk_out = trunk_net(x_trunk, final_act=True)
            output = torch.einsum('ij,ij->i', branch_out, trunk_out) + bias
            outputs.append(output)
        return torch.vstack(outputs).T


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
    model = DeepONet(n_input_branch=50, n_input_trunk=4, layer_width=64, n_layer=2, n_output=4)
    batch_size = 32
    us = torch.rand(size=(batch_size, 50))
    zs = torch.rand(size=(batch_size, 4))
    ps = torch.rand(size=(batch_size, 4))
    p_hats = model(us, zs)
