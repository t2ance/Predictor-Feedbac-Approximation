from typing import Literal

import numpy as np
import torch
from deepxde.nn.pytorch import DeepONet
from numpy import ndarray
from scipy.integrate import odeint

from config import DatasetConfig
from model import FNOProjection
from utils import pad_leading_zeros


class DynamicSystem:
    '''
    The dynamic system with constant C
    '''

    def __init__(self, Z0: ndarray, dataset_config: DatasetConfig,
                 method: Literal['explict', 'numerical', 'no', 'numerical_no'] = None):
        self.method = method
        self.delay = dataset_config.delay
        self.n_point_delay = dataset_config.n_point_delay
        self.Z0 = np.array(Z0)
        self.c = dataset_config.system_c
        self.n = dataset_config.system_n
        self.ts = dataset_config.ts
        self.sequence = iter(range(dataset_config.n_point))
        self.n_point = dataset_config.n_point
        self.dataset_config = dataset_config
        self.dt = dataset_config.dt
        self.init()

    def init(self):
        self.U = np.zeros(self.n_point)
        self.Z = np.zeros((self.n_point, 2))
        self.P = np.zeros((self.n_point, 2))
        self.P_compare = np.zeros((self.n_point, 2))
        self.Z[self.n_point_delay, :] = self.Z0

    def step(self, model=None):
        t_i = next(self.sequence)
        t_minus_D_i = max(t_i - self.n_point_delay, 0)
        t = self.ts[t_i]
        if self.method == 'explict':
            self.U[t_i] = self.U_explict(t)
            if t_i > self.n_point_delay:
                self.Z[t_i, :] = self.Z_explicit(t)
        elif self.method == 'numerical':
            if t_i > self.n_point_delay:
                self.Z[t_i, :] = odeint(self.dynamic, self.Z[t_i - 1, :], [self.ts[t_i - 1], self.ts[t_i]],
                                        args=(self.U[t_minus_D_i - 1],))[1]
                Z_t = self.Z[t_i, :]
            else:
                Z_t = self.Z0
            self.P[t_i, :] = predict_integral_general(f=self.dynamic, Z_t=Z_t, P_D=self.P[t_minus_D_i:t_i],
                                                      U_D=self.U[t_minus_D_i:t_i], dt=self.dt,
                                                      t=t) + self.dataset_config.noise()
            if t_i > self.n_point_delay:
                self.U[t_i] = self.kappa(self.P[t_i, :])
        elif self.method == 'no':
            if t_i > self.n_point_delay:
                self.Z[t_i, :] = \
                    odeint(self.dynamic, self.Z[t_i - 1, :], [self.ts[t_i - 1], self.ts[t_i]],
                           args=(self.U[t_minus_D_i - 1],))[1]
                Z_t = self.Z[t_i, :]
            else:
                Z_t = self.Z0
            self.P[t_i, :] = predict_neural_operator(model=model, U_D=pad_leading_zeros(segment=self.U[t_minus_D_i:t_i],
                                                                                        length=self.n_point_delay),
                                                     Z_t=Z_t, t=t)
            if t_i > self.n_point_delay:
                self.U[t_i] = self.kappa(self.P[t_i, :])
        elif self.method == 'numerical_no':
            if t_i > self.n_point_delay:
                self.Z[t_i, :] = odeint(self.dynamic, self.Z[t_i - 1, :], [self.ts[t_i - 1], self.ts[t_i]],
                                        args=(self.U[t_minus_D_i - 1],))[1]
                Z_t = self.Z[t_i, :]
            else:
                Z_t = self.Z0
            self.P[t_i, :] = predict_integral_general(f=self.dynamic, Z_t=Z_t, P_D=self.P[t_minus_D_i:t_i],
                                                      U_D=self.U[t_minus_D_i:t_i], dt=self.dt,
                                                      t=t) + self.dataset_config.noise()
            self.P_compare[t_i, :] = predict_neural_operator(
                model=model, U_D=pad_leading_zeros(segment=self.U[t_minus_D_i:t_i], length=self.n_point_delay), Z_t=Z_t,
                t=t)
            if t_i > self.n_point_delay:
                self.U[t_i] = self.kappa(self.P[t_i, :])
        else:
            raise NotImplementedError()

    @staticmethod
    def dynamic_static(Z_t, t, U_delay, c=1., n=2.):
        Z1_t = Z_t[0]
        Z2_t = Z_t[1]
        Z1_t_dot = Z2_t - c * Z2_t ** n * U_delay
        Z2_t_dot = U_delay
        return np.array([Z1_t_dot, Z2_t_dot])

    def dynamic(self, Z_t, t, U_delay):
        return DynamicSystem.dynamic_static(Z_t, t, U_delay, self.c, self.n)

    @staticmethod
    def kappa_static(Z_t, c=1., n=2.):
        Z1 = Z_t[0]
        Z2 = Z_t[1]
        return -Z1 - 2 * Z2 - c / (n + 1) * Z2 ** (n + 1)

    def kappa(self, Z_t):
        return DynamicSystem.kappa_static(Z_t, self.c, self.n)

    def U_explict(self, t):
        assert self.c == 1
        assert self.n == 2
        z1_0, z2_0 = self.Z0
        if t >= 0:
            term1 = z1_0 + (2 + self.delay) * z2_0 + (1 / 3) * z2_0 ** 3
            term2 = z1_0 + (1 + self.delay) * z2_0 + (1 / 3) * z2_0 ** 3
            u_t = -np.exp(self.delay - t) * (term1 + (self.delay - t) * term2)
            return u_t
        elif t >= -self.delay:
            return 0
        else:
            raise NotImplementedError()

    def Z_explicit(self, t):
        assert self.c == 1
        assert self.n == 2
        if t < 0:
            raise NotImplementedError()
        if t < self.delay:
            return self.Z0[0] + self.Z0[1] * t, self.Z0[1]
        z1_0 = self.Z0[0]
        z2_0 = self.Z0[1]
        z2_D = self.Z0[1]
        middle_term = z1_0 + self.delay * z2_0 + (1 / 3) * z2_0 ** 3
        term1 = np.exp(self.delay - t) * ((1 + t - self.delay) * middle_term + (t - self.delay) * z2_D)
        term2 = - (1 / 3) * np.exp(3 * (self.delay - t)) * (
                (self.delay - t) * middle_term + (1 - t + self.delay) * z2_D) ** 3
        Z1 = term1 + term2

        Z2 = np.exp(self.delay - t) * ((self.delay - t) * middle_term + (1 - t + self.delay) * z2_D)
        return Z1, Z2


def predict_neural_operator(model, U_D, Z_t, t):
    u_tensor = torch.tensor(U_D, dtype=torch.float32).view(1, -1)
    z_tensor = torch.tensor(Z_t, dtype=torch.float32).view(1, -1)
    inputs = [torch.cat([z_tensor, u_tensor], dim=1), torch.tensor(t, dtype=torch.float32).view(1, -1)]
    if isinstance(model, DeepONet):
        outputs = model(inputs)
    else:
        outputs = model(inputs[0])
    return outputs.to('cpu').detach().numpy()[0]


def predict_integral(Z_t, n_point_delay: int, n_state: int, dt: float, U_D: np.ndarray, dynamic):
    P_D = np.zeros((n_point_delay, n_state))
    P_D[0, :] = Z_t
    for j in range(n_point_delay - 1):
        P_D[j + 1, :] = P_D[j, :] + dt * dynamic(P_D[j, :], j * dt, U_D[j])
    p = predict_integral_general(f=dynamic, Z_t=Z_t, P_D=P_D, U_D=U_D, dt=dt, t=dt * n_point_delay)
    return p


def predict_integral_general(f, Z_t, P_D, U_D, dt, t):
    assert len(P_D) == len(U_D)
    integral = sum([f(p, t, u) for p, u in zip(P_D, U_D)]) * dt
    return integral + Z_t


def ode_forward(Z_t, Z_t_dot, dt):
    return Z_t + Z_t_dot * dt


def predict_integral_explict(t, delay, Z1_t, Z2_t, U_D, ts_D, dt):
    assert len(ts_D) == len(U_D)
    term1 = sum(U_D) * dt
    term2 = sum([(t - theta) * u * dt for u, theta in zip(U_D, ts_D)])
    P1_t = Z1_t + delay * Z2_t + term2 - Z2_t ** 2 * term1 - Z2_t * term1 ** 2
    P2_t = Z2_t + term1
    return P1_t, P2_t


if __name__ == '__main__':
    ...
