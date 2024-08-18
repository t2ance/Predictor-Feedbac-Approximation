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
    def __init__(self, input_size, layer_width, num_layers, output_size):
        super(GRUNet, self).__init__()
        self.hidden_size = layer_width * input_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, self.hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, output_size)
        self.mse_loss = torch.nn.MSELoss()
        self.hidden = None

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        if self.hidden is None:
            self.hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        if x.ndim == 2:
            x = x[:, None, :]

        out, hidden = self.gru(x, self.hidden)
        # self.hidden = hidden.detach()  
        self.hidden = hidden

        out = self.fc(out[:, -1, :])
        if labels is None:
            return out
        return out, self.mse_loss(out, labels)

    def reset_state(self):
        self.hidden = None


class LSTMNet(nn.Module):
    def __init__(self, input_size, layer_width, num_layers, output_size):
        super(LSTMNet, self).__init__()
        self.hidden_size = layer_width * input_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_size, output_size)

        self.hidden = None
        self.cell = None
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        if self.hidden is None or self.cell is None:
            self.hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            self.cell = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        if x.ndim == 2:
            x = x[:, None, :]
        out, (hidden, cell) = self.lstm(x, (self.hidden, self.cell))
        self.hidden = hidden.detach()
        self.cell = cell.detach()

        out = self.fc(out[:, -1, :])
        if labels is None:
            return out
        return out, self.mse_loss(out, labels)

    def reset_state(self):
        self.hidden = None
        self.cell = None


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


class FNOProjectionGRU(torch.nn.Module):
    def __init__(self, n_modes_height: int, hidden_channels: int, fno_n_layers: int, gru_n_layers: int,
                 gru_layer_width: int, n_state: int, fno = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if fno is None:
            fno = FNOProjection(n_modes_height=n_modes_height, hidden_channels=hidden_channels, n_state=n_state,
                                n_layers=fno_n_layers)
        else:
            print('Pretrained FNO model loaded to FNO-GRU')
        self.fno = fno
        self.gru = GRUNet(input_size=n_state, layer_width=gru_layer_width, num_layers=gru_n_layers,
                          output_size=n_state)
        with torch.no_grad():
            for name, param in self.gru.named_parameters():
                param.data.fill_(0.0)
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor, label: torch.Tensor = None):
        x = self.fno(x)
        x = self.gru(x) + x
        if label is None:
            return x
        return x, self.mse_loss(x, label)

    def reset_state(self):
        self.gru.reset_state()


class FNOProjection(torch.nn.Module):
    def __init__(self, n_modes_height: int, hidden_channels: int, n_layers: int, n_state: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_state = n_state
        self.mse_loss = torch.nn.MSELoss()
        self.n_modes_height = n_modes_height
        self.fno = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                         in_channels=2,
                         # in_channels=n_state + 1,
                         out_channels=1)
        self.projection = torch.nn.Linear(in_features=n_modes_height * 2, out_features=n_state)
        self.linear_decoder = LinearDecoder(n_state, 1, n_modes_height)
        self.linear_functional = LinearFunctional(1, n_state, n_modes_height)

    def forward(self, x: torch.Tensor, label: torch.Tensor = None):
        # u = x[:, self.n_state:]
        # z = x[:, :self.n_state]
        # U_expanded = u.unsqueeze(1)
        # Z_expanded = z.unsqueeze(2)
        # broadcast_sum = U_expanded + Z_expanded
        # x_in = torch.cat((U_expanded, broadcast_sum), dim=1)
        # x_in = torch.einsum('bi,bj->bij', torch.cat([z, torch.ones_like(z)[:, 0:1]], dim=1), u)

        # x_in = x.unsqueeze(-2)
        # x = self.fno(x_in)
        # x = fft_transform(x.squeeze(1), self.n_modes_height)
        # x = self.projection(x)

        u = x[:, self.n_state:].unsqueeze(-2)
        z = x[:, :self.n_state]
        z = self.linear_decoder(z, u.shape[-1])
        x = torch.concatenate([u, z], 1)
        x = self.fno(x)
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


class FNOTwoStage(torch.nn.Module):
    def __init__(self, n_modes_height: int, hidden_channels: int, n_state: int, n_layers: int, dt: float, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.fno = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                         in_channels=n_state + 1, out_channels=n_state)
        self.integral_net_p = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                                    in_channels=n_state, out_channels=n_state)
        self.integral_net_u = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                                    in_channels=1, out_channels=n_state)
        self.integral_net_p_u = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                                      in_channels=n_state, out_channels=n_state)
        self.dt = dt

    def forward(self, z_u: torch.Tensor, label: torch.Tensor = None):
        U = z_u[:, 2:]
        Z = z_u[:, :2]
        # out = self.fno(Z_U.unsqueeze(-2)).squeeze(1)

        P = self.fno(torch.einsum('bi,bj->bij', torch.cat([Z, torch.ones_like(Z)[:, 0:1]], dim=1), U))
        # find p
        # P = torch.zeros((Z_U.shape[0], Z.shape[1], U.shape[1]))
        # P[:, :, 0] = Z
        # for j in range(U.shape[1] - 1):
        #     Z2_t = P[:, 1, j]
        #     Z1_t_dot = Z2_t - Z2_t ** 2 * U[:, j]
        #     Z2_t_dot = U[:, j]
        #     partial = torch.hstack([Z1_t_dot.reshape(-1, 1), Z2_t_dot.reshape(-1, 1)]) * self.dt
        #     P[:, :, j + 1] = P[:, :, j] + partial

        # predict_integral_general
        # integral = torch.zeros_like(Z_U[:, :2])
        # for i in range(P.shape[-1]):
        #     Z2_t = P[:, 1, i]
        #     U_delay = U[:, i]
        #     Z1_t_dot = Z2_t - Z2_t ** 2 * U_delay
        #     Z2_t_dot = U_delay
        #     partial = torch.hstack([Z1_t_dot.reshape(-1, 1), Z2_t_dot.reshape(-1, 1)]) * self.dt
        #     integral += partial
        p_ = self.integral_net_p(P)
        u_ = self.integral_net_u(U.unsqueeze(1))
        p_u = self.integral_net_p_u(p_ + u_)
        x = p_u.sum(axis=-1) * self.dt + Z
        if label is None:
            return x
        return x, self.mse_loss(x, label)


if __name__ == '__main__':
    ...
