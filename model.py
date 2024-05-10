import numpy as np
import torch
from neuralop.models import FNO1d


class FNOProjection(torch.nn.Module):
    def __init__(self, n_modes_height: int, hidden_channels: int, n_layers: int,
                 dt: float, n_state: int, n_point_delay: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.fno = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
        #                  in_channels=1, out_channels=1)
        # self.projection = torch.nn.Linear(in_features=n_state + n_point_delay, out_features=n_state)
        self.fno = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                         in_channels=n_state, out_channels=n_state)
        self.integral_net_p = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                                    in_channels=n_state, out_channels=n_state)
        self.integral_net_u = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                                    in_channels=1, out_channels=n_state)
        self.integral_net_p_u = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                                      in_channels=n_state, out_channels=n_state)
        self.fno = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                         in_channels=1, out_channels=1)
        self.projection = torch.nn.Linear(in_features=n_state + n_point_delay, out_features=n_state)
        self.dt = dt

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-2)
        x = self.fno(x)
        x = self.projection(x)
        x = x.squeeze(-2)
        return x

    # def forward(self, Z_U: torch.Tensor):
    #     U = Z_U[:, 2:]
    #     Z = Z_U[:, :2]
    #     # out = self.fno(Z_U.unsqueeze(-2)).squeeze(1)
    #     P = self.fno(torch.einsum('bi,bj->bij', Z, U))
    #     # find p
    #     # P = torch.zeros((Z_U.shape[0], Z.shape[1], U.shape[1]))
    #     # P[:, :, 0] = Z
    #     # for j in range(U.shape[1] - 1):
    #     #     Z2_t = P[:, 1, j]
    #     #     Z1_t_dot = Z2_t - Z2_t ** 2 * U[:, j]
    #     #     Z2_t_dot = U[:, j]
    #     #     partial = torch.hstack([Z1_t_dot.reshape(-1, 1), Z2_t_dot.reshape(-1, 1)]) * self.dt
    #     #     P[:, :, j + 1] = P[:, :, j] + partial
    #
    #     # predict_integral_general
    #     # integral = torch.zeros_like(Z_U[:, :2])
    #     # for i in range(P.shape[-1]):
    #     #     Z2_t = P[:, 1, i]
    #     #     U_delay = U[:, i]
    #     #     Z1_t_dot = Z2_t - Z2_t ** 2 * U_delay
    #     #     Z2_t_dot = U_delay
    #     #     partial = torch.hstack([Z1_t_dot.reshape(-1, 1), Z2_t_dot.reshape(-1, 1)]) * self.dt
    #     #     integral += partial
    #     p_ = self.integral_net_p(P)
    #     u_ = self.integral_net_u(U.unsqueeze(1))
    #     p_u = self.integral_net_p_u(p_ + u_)
    #
    #     return p_u.sum(axis=-1) * self.dt + Z


class FNOSum(torch.nn.Module):
    def __init__(self, n_modes_height: int, hidden_channels: int, in_features: int, n_layers: int, out_features: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fno = FNO1d(n_modes_height=n_modes_height, n_layers=n_layers, hidden_channels=hidden_channels,
                         in_channels=1, out_channels=1)

    def forward(self, z_u: torch.Tensor):
        z_u = z_u.unsqueeze(-2)
        p = self.fno(z_u)
        # p = predict_integral_general(f=dynamic, Z_t=Z_t, P_D=P_D, U_D=U_D, dt=dt, t=dt * n_point_delay)
        z_u = self.projection(z_u)
        z_u = z_u.squeeze(-2)
        return z_u


if __name__ == '__main__':
    ...
