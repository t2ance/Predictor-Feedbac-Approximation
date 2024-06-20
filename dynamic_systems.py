from abc import abstractmethod

import numpy as np
import torch
from scipy.integrate import simps


class DynamicSystem:
    def __init__(self, delay):
        super().__init__()
        self.delay = delay

    def U_explicit(self, t, Z0):
        raise NotImplementedError()

    def Z_explicit(self, t, Z0):
        raise NotImplementedError()

    @abstractmethod
    def name(self):
        ...

    def n_state(self):
        raise NotImplementedError()

    @abstractmethod
    def dynamic(self, Z_t, t, U_delay):
        ...

    @abstractmethod
    def kappa(self, Z_t):
        ...

    @abstractmethod
    def dynamic_tensor_batched1(self, Z_t, t, U_delay):
        ...

    @abstractmethod
    def dynamic_tensor_batched2(self, Z_t, t, U_delay):
        ...


class DynamicSystem1(DynamicSystem):
    '''
    The dynamic system with constant c & n
    '''

    @property
    def name(self):
        return 's1'

    @property
    def n_state(self):
        return 2

    def __init__(self, c, n, delay):
        super().__init__(delay)
        self.c = c
        self.n = n

    def dynamic(self, Z_t, t, U_delay):
        if len(Z_t.shape) < 2:
            Z1_t, Z2_t = Z_t
            Z1_t_dot = Z2_t - self.c * Z2_t ** self.n * U_delay
            Z2_t_dot = U_delay
            return np.array([Z1_t_dot, Z2_t_dot])
        else:
            Z1_t, Z2_t = Z_t[:, 0], Z_t[:, 1]
            Z1_t_dot = Z2_t - self.c * Z2_t ** self.n * U_delay
            Z2_t_dot = U_delay
            return np.vstack([Z1_t_dot, Z2_t_dot]).T

    def dynamic_tensor_batched1(self, Z_t, t, U_delay):
        Z1_t = Z_t[:, 0]
        Z2_t = Z_t[:, 1]
        Z1_t_dot = Z2_t - self.c * Z2_t ** self.n * U_delay
        Z2_t_dot = U_delay
        return torch.stack([Z1_t_dot, Z2_t_dot], dim=1)

    def dynamic_tensor_batched2(self, Z_t, t, U_delay):
        Z1_t = Z_t[:, :, 0]
        Z2_t = Z_t[:, :, 1]
        Z1_t_dot = Z2_t - self.c * Z2_t ** self.n * U_delay
        Z2_t_dot = U_delay
        return torch.stack([Z1_t_dot, Z2_t_dot], dim=1)

    def kappa(self, Z_t):
        Z1 = Z_t[0]
        Z2 = Z_t[1]
        return -Z1 - 2 * Z2 - self.c / (self.n + 1) * Z2 ** (self.n + 1)

    def U_explicit(self, t, Z0):
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


class DynamicSystem2(DynamicSystem):
    @property
    def name(self):
        return 's2'

    @property
    def n_state(self):
        return 3

    def __init__(self, delay):
        super().__init__(delay)

    def dynamic(self, Z_t, t, U_delay):
        if len(Z_t.shape) < 2:
            Z1_t, Z2_t, Z3_t = Z_t
            Z1_t_dot = Z2_t + Z3_t ** 2
            Z2_t_dot = Z3_t + Z3_t * U_delay
            Z3_t_dot = U_delay
            return np.array([Z1_t_dot, Z2_t_dot, Z3_t_dot])
        else:
            Z1_t, Z2_t, Z3_t = Z_t[:, 0], Z_t[:, 1], Z_t[:, 2]
            Z1_t_dot = Z2_t + Z3_t ** 2
            Z2_t_dot = Z3_t + Z3_t * U_delay
            Z3_t_dot = U_delay
            return np.vstack([Z1_t_dot, Z2_t_dot, Z3_t_dot]).T

    def dynamic_tensor_batched1(self, Z_t, t, U_delay):
        Z1_t = Z_t[:, 0]
        Z2_t = Z_t[:, 1]
        Z3_t = Z_t[:, 2]
        Z1_t_dot = Z2_t + Z3_t ** 2
        Z2_t_dot = Z3_t + Z3_t * U_delay
        Z3_t_dot = U_delay
        return torch.stack([Z1_t_dot, Z2_t_dot, Z3_t_dot], dim=1)

    def dynamic_tensor_batched2(self, Z_t, t, U_delay):
        Z1_t = Z_t[:, :, 0]
        Z2_t = Z_t[:, :, 1]
        Z3_t = Z_t[:, :, 2]
        Z1_t_dot = Z2_t + Z3_t ** 2
        Z2_t_dot = Z3_t + Z3_t * U_delay
        Z3_t_dot = U_delay
        return torch.stack([Z1_t_dot, Z2_t_dot, Z3_t_dot], dim=1)

    def kappa(self, Z_t):
        P1 = Z_t[0]
        P2 = Z_t[1]
        P3 = Z_t[2]
        return (
                -P1 - 3 * P2 - 3 * P3 - 3 / 8 * P2 ** 2
                + 3 / 4 * P3 * (-P1 - 2 * P2 + 1 / 2 * P3
                                + P2 * P3 / 2 + 5 / 8 * P3 ** 2 - 1 / 4 * P3 ** 3
                                - 3 / 8 * (P2 - P3 ** 2 / 2) ** 2)
        )


class InvertedPendulum(DynamicSystem):
    def __init__(self, delay: float, M=1.0, m=0.1, l=1.0, g=9.81, desired_poles=[-2, -2.5, -3, -3.5]):
        self.M = M
        self.m = m
        self.l = l
        self.g = g
        self.desired_poles = desired_poles
        self.A = np.array([
            [0, 1, 0, 0],
            [(self.M + self.m) * self.g / (self.M * self.l), 0, 0, 0],
            [0, 0, 0, 1],
            [-self.m * self.g / self.M, 0, 0, 0]
        ])
        self.B = np.array([
            [0],
            [-1 / (self.M * self.l)],
            [0],
            [1 / self.M]
        ])
        self.K = self.calculate_feedback_gain()
        super().__init__(delay)

    @property
    def n_state(self):
        return 4

    def name(self):
        return 's3'

    def calculate_feedback_gain(self):
        from scipy.signal import place_poles
        result = place_poles(self.A, self.B, self.desired_poles)
        K = result.gain_matrix[0]
        return K

    def dynamic(self, Z_t, t, U_delay):
        if len(Z_t.shape) < 2:
            Z_dot = np.dot(self.A, Z_t) + self.B.flatten() * U_delay
            return Z_dot
        else:
            Z_dot = np.dot(self.A, Z_t.T) + self.B * U_delay
            return Z_dot.T

    def kappa(self, Z_t):
        u = -np.dot(self.K, Z_t)
        return u

    def dynamic_tensor_batched1(self, Z_t, t, U_delay):
        raise NotImplementedError()

    def dynamic_tensor_batched2(self, Z_t, t, U_delay):
        raise NotImplementedError()


class VanDerPolOscillator(DynamicSystem):
    def name(self):
        return 's4'

    @property
    def n_state(self):
        return 2

    def __init__(self, delay: float, mu=1.0, desired_poles=[-2, -3]):
        super().__init__(delay)
        self.mu = mu
        self.desired_poles = desired_poles
        self.A = np.array([[0, 1], [0, -1]])
        self.B = np.array([[0], [1]])
        self.K = self.calculate_feedback_gain()

    def calculate_feedback_gain(self):
        from scipy.signal import place_poles
        result = place_poles(self.A, self.B, self.desired_poles)
        K = result.gain_matrix[0]
        return K

    def dynamic(self, Z_t, t, U_delay):
        if len(Z_t.shape) < 2:
            x1, x2 = Z_t
            x1_dot = x2
            x2_dot = self.mu * (1 - x1 ** 2) * x2 - x1 + U_delay
            return np.array([x1_dot, x2_dot])
        else:
            x1, x2 = Z_t[:, 0], Z_t[:, 1]
            x1_dot = x2
            x2_dot = self.mu * (1 - x1 ** 2) * x2 - x1 + U_delay
            return np.vstack([x1_dot, x2_dot]).T

    def kappa(self, Z_t):
        u = -np.dot(self.K, Z_t)
        return u

    def dynamic_tensor_batched1(self, Z_t, t, U_delay):
        raise NotImplementedError()

    def dynamic_tensor_batched2(self, Z_t, t, U_delay):
        raise NotImplementedError()


def solve_integral_nn(model, U_D, Z_t):
    device = next(model.parameters()).device
    u_tensor = torch.tensor(U_D, dtype=torch.float32, device=device).view(1, -1)
    z_tensor = torch.tensor(Z_t, dtype=torch.float32, device=device).view(1, -1)
    outputs = model(torch.cat([z_tensor, u_tensor], dim=1))
    return outputs.to('cpu').detach().numpy()[0]


def solve_integral_ci_nn(mapie_regressors, U_D, Z_t):
    device = mapie_regressors.device
    u_tensor = torch.tensor(U_D, dtype=torch.float32, device=device).view(1, -1)
    z_tensor = torch.tensor(Z_t, dtype=torch.float32, device=device).view(1, -1)
    inputs = torch.cat([z_tensor, u_tensor], dim=1)
    return mapie_regressors.predict(inputs)


def solve_integral(Z_t, P_D, U_D, t: float, dataset_config):
    assert len(P_D) == len(U_D)
    system = dataset_config.system
    dt = dataset_config.dt
    f = system.dynamic
    n_state = system.n_state
    n_points = len(P_D)
    assert n_points <= dataset_config.n_point_delay
    ts = np.linspace(t - n_points * dt, t - dt, n_points)
    if dataset_config.integral_method == 'rectangle':
        return solve_integral_rectangle(f=f, Z_t=Z_t, P_D=P_D, U_D=U_D, ts=ts, dt=dt)
    elif dataset_config.integral_method == 'trapezoidal':
        return solve_integral_trapezoidal(f=f, Z_t=Z_t, P_D=P_D, U_D=U_D, ts=ts, dt=dt)
    elif dataset_config.integral_method == 'simpson':
        return solve_integral_simpson(f=f, Z_t=Z_t, P_D=P_D, U_D=U_D, ts=ts, dt=dt)
    elif dataset_config.integral_method == 'eular':
        return solve_integral_eular(Z_t=Z_t, n_points=n_points, n_state=n_state, dt=dt, U_D=U_D, f=f)
    elif dataset_config.integral_method == 'successive':
        return solve_integral_successive(Z_t=Z_t, n_points=n_points, n_state=n_state, dt=dt, U_D=U_D, f=f,
                                         n_iterations=dataset_config.successive_approximation_n_iteration)
    else:
        raise NotImplementedError()


def solve_integral_eular(Z_t, n_points: int, n_state: int, dt: float, U_D: np.ndarray, f):
    if isinstance(Z_t, np.ndarray):
        P_D = np.zeros((n_points + 1, n_state))
    elif isinstance(Z_t, torch.Tensor):
        P_D = torch.zeros((n_points + 1, n_state))
    else:
        raise NotImplementedError()
    P_D[0, :] = Z_t
    for j in range(n_points):
        P_D[j + 1, :] = P_D[j, :] + dt * f(P_D[j, :], j * dt, U_D[j])
    return P_D[-1]


def solve_integral_successive(Z_t, n_points: int, n_state: int, dt: float, U_D: np.ndarray, f, n_iterations: int):
    assert n_iterations >= 0
    if isinstance(Z_t, np.ndarray):
        P_D = np.zeros((n_iterations + 1, n_points + 1, n_state))
    elif isinstance(Z_t, torch.Tensor):
        P_D = torch.zeros((n_iterations + 1, n_points + 1, n_state))
    else:
        raise NotImplementedError()
    P_D[0, :, :] = Z_t

    for n in range(n_iterations):
        for j in range(n_points):
            P_D[n + 1, j + 1, :] = Z_t + np.sum([dt * f(P_D[n, k, :], k * dt, U_D[k]) for k in range(j + 1)], axis=0)
    return P_D[-1, -1, :]


def solve_integral_rectangle(f, Z_t, P_D, U_D, ts, dt: float):
    assert len(P_D) == len(U_D)
    if len(P_D) == 0:
        return 0
    integrand_values = f(np.array(P_D), np.array(ts), np.array(U_D))
    integral = integrand_values.sum(0) * dt
    return integral + Z_t


def solve_integral_trapezoidal(f, Z_t, P_D, U_D, ts, dt: float):
    assert len(P_D) == len(U_D)
    if len(P_D) == 0:
        return 0
    integrand_values = f(np.array(P_D), np.array(ts), np.array(U_D))
    t_values = np.arange(0, len(integrand_values)) * dt
    integral = np.trapz(integrand_values, t_values)
    return integral + Z_t


def solve_integral_simpson(f, Z_t, P_D, U_D, ts, dt: float):
    assert len(P_D) == len(U_D)
    if len(P_D) == 0:
        return 0
    integrand_values = f(np.array(P_D), np.array(ts), np.array(U_D))
    t_values = np.arange(0, len(integrand_values)) * dt
    integral = simps(integrand_values, t_values)
    return integral + Z_t


def ode_forward(Z_t, Z_t_dot, dt):
    return Z_t + Z_t_dot * dt


def solve_integral_equation_explicit(t, delay, Z1_t, Z2_t, U_D, ts_D, dt):
    assert len(ts_D) == len(U_D)
    term1 = sum(U_D) * dt
    term2 = sum([(t - theta) * u * dt for u, theta in zip(U_D, ts_D)])
    P1_t = Z1_t + delay * Z2_t + term2 - Z2_t ** 2 * term1 - Z2_t * term1 ** 2
    P2_t = Z2_t + term1
    return P1_t, P2_t


if __name__ == '__main__':
    from config import DatasetConfig

    dataset_config = DatasetConfig()
    solve_integral_eular(np.array([0, 1]), dataset_config.n_point_delay, dataset_config.n_state, dataset_config.dt,
                         np.random.randn(dataset_config.n_point_delay),
                         f=dataset_config.system.dynamic)
    ...
