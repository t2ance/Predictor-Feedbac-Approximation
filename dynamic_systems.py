import numpy as np
import torch
from deepxde.nn.pytorch import DeepONet


class DynamicSystem:
    '''
    The dynamic system with constant c & n
    '''

    def __init__(self, c, n, delay):
        self.c = c
        self.n = n
        self.delay = delay

    def dynamic(self, Z_t, t, U_delay):
        Z1_t = Z_t[0]
        Z2_t = Z_t[1]
        Z1_t_dot = Z2_t - self.c * Z2_t ** self.n * U_delay
        Z2_t_dot = U_delay
        return np.array([Z1_t_dot, Z2_t_dot])

    def dynamic_tensor_batched(self, Z_t, t, U_delay):
        Z1_t = Z_t[0]
        Z2_t = Z_t[1]
        Z1_t_dot = Z2_t - self.c * Z2_t ** self.n * U_delay
        Z2_t_dot = U_delay
        return torch.tensor([Z1_t_dot, Z2_t_dot])

    def kappa(self, Z_t):
        Z1 = Z_t[0]
        Z2 = Z_t[1]
        return -Z1 - 2 * Z2 - self.c / (self.n + 1) * Z2 ** (self.n + 1)

    def U_explict(self, t, Z0):
        assert self.c == 1
        assert self.n == 2
        z1_0, z2_0 = Z0
        if t >= 0:
            term1 = z1_0 + (2 + self.delay) * z2_0 + (1 / 3) * z2_0 ** 3
            term2 = z1_0 + (1 + self.delay) * z2_0 + (1 / 3) * z2_0 ** 3
            u_t = -np.exp(self.delay - t) * (term1 + (self.delay - t) * term2)
            return u_t
        elif t >= -self.delay:
            return 0
        else:
            raise NotImplementedError()

    def Z_explicit(self, t, Z0):
        assert self.c == 1
        assert self.n == 2
        if t < 0:
            raise NotImplementedError()
        if t < self.delay:
            return Z0[0] + Z0[1] * t, Z0[1]
        z1_0 = Z0[0]
        z2_0 = Z0[1]
        z2_D = Z0[1]
        middle_term = z1_0 + self.delay * z2_0 + (1 / 3) * z2_0 ** 3
        term1 = np.exp(self.delay - t) * ((1 + t - self.delay) * middle_term + (t - self.delay) * z2_D)
        term2 = - (1 / 3) * np.exp(3 * (self.delay - t)) * (
                (self.delay - t) * middle_term + (1 - t + self.delay) * z2_D) ** 3
        Z1 = term1 + term2

        Z2 = np.exp(self.delay - t) * ((self.delay - t) * middle_term + (1 - t + self.delay) * z2_D)
        return Z1, Z2


def predict_neural_operator(model, U_D, Z_t, t):
    device = next(model.parameters()).device
    u_tensor = torch.tensor(U_D, dtype=torch.float32, device=device).view(1, -1)
    z_tensor = torch.tensor(Z_t, dtype=torch.float32, device=device).view(1, -1)
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
    # u = DynamicSystem(Z0=np.array([0, 1]), dataset_config=DatasetConfig()).U_explict(0)
    # print(u)
    ...
