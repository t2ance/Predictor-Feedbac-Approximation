from typing import Callable

import torch
from neuralop.models import FNO1d


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


class FNOProjection(torch.nn.Module):
    def __init__(self, n_modes_height: int, hidden_channels: int, n_layers: int, n_state: int, n_point_delay: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse_loss = torch.nn.MSELoss()
        self.fno = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                         in_channels=1, out_channels=1)
        in_features = n_state + n_point_delay
        out_features = n_state
        self.projection = torch.nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x: torch.Tensor, label: torch.Tensor = None):
        x = x.unsqueeze(-2)
        x = self.fno(x)
        x = self.projection(x)
        x = x.squeeze(-2)
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


class FNOTwoStage2(torch.nn.Module):
    def __init__(self, n_modes_height: int, hidden_channels: int, n_state: int, n_layers: int, dt: float, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.fno = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                         in_channels=n_state, out_channels=n_state)
        self.integral_net_p = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                                    in_channels=n_state, out_channels=n_state)
        self.integral_net_u = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                                    in_channels=1, out_channels=n_state)
        self.integral_net_p_u = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                                      in_channels=n_state, out_channels=n_state)

        self.dt = dt

    def forward(self, z_u: torch.Tensor):
        U = z_u[:, 2:]
        Z = z_u[:, :2]
        # out = self.fno(Z_U.unsqueeze(-2)).squeeze(1)
        P = self.fno(torch.einsum('bi,bj->bij', Z, U))
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

        return p_u.sum(axis=-1) * self.dt + Z


if __name__ == '__main__':
    ...
