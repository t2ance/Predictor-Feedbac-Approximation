from typing import Callable

import numpy as np
import torch
from mapie.regression import MapieRegressor
from neuralop.models import FNO1d
from sklearn.base import RegressorMixin
from torch import nn
from torch.nn import init


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


# Wrapper Models
class MapieRegressors:
    def __init__(self, model, n_output: int, device, alpha=0.1):
        self.model = model
        self.n_output = n_output
        self.mapie_regressors = []
        self.device = device
        self.alpha = alpha
        for output_dim in range(self.n_output):
            self.mapie_regressors.append(
                MapieRegressor(estimator=ConformalPredictionWrapper(model, device=device, output_dim=output_dim),
                               cv='prefit')
            )

    def fit(self, X, y):
        for output_dim in range(self.n_output):
            self.mapie_regressors[output_dim].fit(X[:, 1:], y[:, output_dim])
        return self

    def predict(self, X, alpha: float = None):
        if alpha is None:
            alpha = self.alpha
        y_pis_list = []
        y_pred_list = []
        for output_dim in range(self.n_output):
            y_pred, y_pis = self.mapie_regressors[output_dim].predict(X, alpha=alpha)
            y_pis_list.append(y_pis[0, :, 0])
            y_pred_list.append(y_pred)
        return np.array(y_pis_list), np.array(y_pred_list)


class ConformalPredictionWrapper(RegressorMixin):
    def __init__(self, model, device, output_dim: int = -1):
        self.model = model
        self.device = device
        self.output_dim = output_dim

    def fit(self, X, y):
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(torch.Tensor(X).to(self.device)).detach().cpu().numpy()
        if self.output_dim == -1:
            return predictions
        else:
            return predictions[:, self.output_dim]

    def __sklearn_is_fitted__(self):
        return True


# Generation Models

class BSplineNet(nn.Module):
    def __init__(self, n_state: int, n_knot: int, degree: int, points):
        super(BSplineNet, self).__init__()
        self.points = torch.tensor(points)
        self.n_knot = n_knot
        self.degree = degree
        self.net = nn.Sequential(
            nn.Linear(2 * n_state, 8 * n_state),
            nn.ReLU(),
            nn.Linear(8 * n_state, 8 * n_state),
            nn.ReLU(),
            nn.Linear(8 * n_state, 4 * n_state),
            nn.ReLU(),
            nn.Linear(4 * n_state, n_knot)
        )
        self.knots = torch.linspace(points.min(), points.max(), n_knot + degree + 1)

    def forward(self, x):
        coefficients = self.net(x)

        def b_spline_basis(x, degree, i, knots):
            if degree == 0:
                return ((knots[i] <= x) & (x < knots[i + 1])).float()
            else:
                left = ((x - knots[i]) / (knots[i + degree] - knots[i])) * b_spline_basis(x, degree - 1, i, knots)
                right = ((knots[i + degree + 1] - x) / (knots[i + degree + 1] - knots[i + 1])) \
                        * b_spline_basis(x, degree - 1, i + 1, knots)
                return left + right

        basis = torch.stack([b_spline_basis(self.points, self.degree, i, self.knots) for i in range(self.n_knot)],
                            dim=0).to(dtype=coefficients.dtype)
        series = torch.matmul(coefficients, basis)
        return series


class ChebyshevNet(nn.Module):
    def __init__(self, n_state: int, n_terms: int, points):
        assert points.max().item() <= 1 and points.min().item() >= 0
        super(ChebyshevNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * n_state, 8 * n_state),
            nn.ReLU(),
            nn.Linear(8 * n_state, 4 * n_state),
            nn.ReLU(),
            nn.Linear(4 * n_state, n_terms)
        )
        self.points = torch.tensor(points)  # ndarray of points where we want to evaluate the function
        self.n_terms = n_terms

    def forward(self, x):
        coefficients = self.net(x)

        # Define Chebyshev polynomial function
        def chebyshev_polynomial(n, x):
            return torch.cos(n * torch.acos(x)).to(dtype=coefficients.dtype)

        # Generate all Chebyshev polynomials up to n_terms
        n_values = torch.arange(self.n_terms, dtype=coefficients.dtype).unsqueeze(1)
        T_n = chebyshev_polynomial(n_values, self.points.unsqueeze(0))

        # Construct the Chebyshev series approximation using matrix multiplication
        series = torch.matmul(coefficients, T_n)

        return series


class FourierNet(nn.Module):
    def __init__(self, n_state: int, n_mode: int, points):
        super(FourierNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2 * n_state, 8 * n_state),
            nn.ReLU(),
            nn.Linear(8 * n_state, 4 * n_state),
            nn.ReLU(),
            nn.Linear(4 * n_state, 2 * n_mode + 1)
        )
        self.points = torch.tensor(points)  # points where we want to evaluate the function

    def forward(self, x):
        coefficients = self.net(x)
        # Construct the Fourier series approximation
        n = (coefficients.shape[1] - 1) // 2
        a0 = coefficients[:, 0] / 2.0

        # Generate the cosine and sine terms for all harmonics
        harmonics = torch.arange(1, n + 1, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
        points_expanded = self.points.unsqueeze(0).unsqueeze(0)

        cos_terms = torch.cos(2 * torch.pi * harmonics * points_expanded)
        sin_terms = torch.sin(2 * torch.pi * harmonics * points_expanded)

        # Extract the cosine and sine coefficients
        cos_coefficients = coefficients[:, 1::2].unsqueeze(2)
        sin_coefficients = coefficients[:, 2::2].unsqueeze(2)

        # Compute the series using broadcasting
        series = a0.unsqueeze(1) + torch.sum(cos_coefficients * cos_terms + sin_coefficients * sin_terms, dim=1)

        return series


class FullyConnectedNet(torch.nn.Module):
    def __init__(self, n_state: int, n_point_delay: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(2 * n_state, 4 * n_point_delay),
            nn.ReLU(),
            nn.Linear(4 * n_point_delay, 4 * n_point_delay),
            nn.ReLU(),
            nn.Linear(4 * n_point_delay, 4 * n_point_delay),
            nn.ReLU(),
            nn.Linear(4 * n_point_delay, n_point_delay)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


# Prediction Models
class QuantileRegressionModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, alpha: float):
        super(QuantileRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim * 2)  # Output two values per dimension
        self.alpha = alpha
        self.quantiles = [(alpha / 2, 1 - alpha / 2)] * output_dim

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(x.size(0), -1, 2)  # Reshape to (batch_size, output_dim, 2)


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
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

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
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

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


class PIFNO(torch.nn.Module):
    def __init__(self, n_modes_height: int, hidden_channels: int, n_layers: int, dt: float, n_state: int,
                 dynamic: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fno = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                         in_channels=1 + n_state, out_channels=n_state)
        self.dt = dt
        self.dynamic = dynamic
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        U = x[:, 2:]
        Z = x[:, :2]
        P = self.fno(torch.einsum('bi,bj->bij', torch.cat([Z, torch.ones_like(Z)[:, 0:1]], dim=1), U))
        # outputs = P[:, :, -1]
        outputs = self.dynamic(P.permute(0, 2, 1), None, U).sum(dim=-1) * self.dt + Z
        if labels is None:
            return outputs
        loss_mse = self.mse_loss(outputs, labels)
        # loss_ie = ((outputs - labels).norm(dim=1) ** 2).sum()
        # loss_ie = loss_ie / len(P)
        return outputs, loss_mse


def fft_transform(input_tensor, target_length):
    fft_result = torch.fft.rfft(input_tensor).real

    current_length = fft_result.shape[1]

    if current_length > target_length:
        output = fft_result[:, :target_length]
    elif current_length < target_length:
        padding = torch.zeros((input_tensor.shape[0], target_length - current_length), device=input_tensor.device)
        output = torch.cat((fft_result, padding), dim=1)
    else:
        output = fft_result

    return output.to(input_tensor.device)


class FNOProjectionGRU(torch.nn.Module):
    def __init__(self, n_modes_height: int, hidden_channels: int, fno_n_layers: int, gru_n_layers: int,
                 gru_layer_width: int, n_state: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fno = FNOProjection(n_modes_height=n_modes_height, hidden_channels=hidden_channels, n_state=n_state,
                                 n_layers=fno_n_layers)
        self.gru = GRUNet(input_size=n_state, hidden_size=n_state * gru_layer_width, num_layers=gru_n_layers,
                          output_size=n_state)
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor, label: torch.Tensor = None):
        x = self.fno(x)
        x = self.gru(x)
        if label is None:
            return x
        return x, self.mse_loss(x, label)

    def reset_state(self):
        self.gru.reset_state()


class FNOProjection(torch.nn.Module):
    def __init__(self, n_modes_height: int, hidden_channels: int, n_layers: int, n_state: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse_loss = torch.nn.MSELoss()
        self.n_modes_height = n_modes_height
        self.fno = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                         in_channels=1, out_channels=1)
        out_features = n_state
        self.projection = torch.nn.Linear(in_features=n_modes_height, out_features=out_features)

    def forward(self, x: torch.Tensor, label: torch.Tensor = None):
        x = x.unsqueeze(-2)
        x = self.fno(x)
        x = fft_transform(x.squeeze(1), self.n_modes_height)
        x = self.projection(x)
        if label is None:
            return x
        return x, self.mse_loss(x, label)


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
