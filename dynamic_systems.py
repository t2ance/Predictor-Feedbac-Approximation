from abc import abstractmethod
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from scipy.integrate import simps

from baxter import BaxterParameters


@dataclass
class IntegralSolution:
    solution: np.ndarray = None
    n_iter: int = None


class DynamicSystem:

    @property
    def name(self):
        raise NotImplementedError()

    @property
    def n_state(self):
        raise NotImplementedError()

    @property
    def n_input(self):
        return 1

    @abstractmethod
    def dynamic(self, Z_t, t, U_delay):
        ...

    @abstractmethod
    def kappa(self, Z_t, t):
        ...


class Unicycle(DynamicSystem):
    @property
    def name(self):
        return 's7'

    @property
    def n_state(self):
        return 3

    @property
    def n_input(self):
        return 2

    def dynamic(self, Z_t, t, U_delay):
        x, y, theta = Z_t
        omega, nu = U_delay
        x_dot = nu * np.cos(theta)
        y_dot = nu * np.sin(theta)
        theta_dot = omega
        return np.array([x_dot, y_dot, theta_dot])

    def kappa(self, Z_t, t):
        x, y, theta = Z_t
        p = x * np.cos(theta) + y * np.sin(theta)
        q = x * np.sin(theta) - y * np.cos(theta)
        omega = -5 * p ** 2 * np.cos(3 * t) - p * q * (1 + 25 * np.cos(3 * t) ** 2) - theta
        nu = -p + 5 * q * (np.sin(3 * t) - np.cos(3 * t)) + q * omega
        return np.array([omega, nu])


class Baxter(DynamicSystem):
    '''
    A dynamic system class for the Baxter robot manipulator with input delay,
    implementing the dynamics based on the equations described in the paper.
    '''

    @property
    def name(self):
        return 's5'

    @property
    def n_input(self):
        return self.dof

    @property
    def n_state(self):
        return self.dof * 2  # dof dimensions for e1 and dof dimensions for e2

    def __init__(self, alpha=None, beta=None, dof: int = 7, f: float = 0.1, magnitude: float = 0.2):
        assert 1 <= dof <= 7
        self.dof = dof
        self.f = f
        self.magnitude = magnitude
        self.alpha = np.eye(dof) if alpha is None else alpha
        self.beta = np.eye(dof) if beta is None else beta
        self.baxter_parameters = BaxterParameters(dof=dof)

    @lru_cache(maxsize=None)
    def G(self, t):
        return self.baxter_parameters.compute_gravity_vector(self.q_des(t))

    @lru_cache(maxsize=None)
    def C(self, t):
        return self.baxter_parameters.compute_coriolis_centrifugal_matrix(self.q_des(t), self.qd_des(t))

    @lru_cache(maxsize=None)
    def M(self, t):
        return self.baxter_parameters.compute_inertia_matrix(self.q_des(t))

    @lru_cache(maxsize=None)
    def q_des(self, t):
        return self.magnitude * np.array(
            [1 * np.sin(self.f * t), 1 * np.cos(self.f * t), 1 * np.sin(self.f * t), 1 * np.cos(self.f * t),
             1 * np.sin(self.f * t), 0,
             0])[:self.dof]

    @lru_cache(maxsize=None)
    def qd_des(self, t):
        return self.magnitude * np.array(
            [self.f * np.cos(self.f * t), -self.f * np.sin(self.f * t), self.f * np.cos(self.f * t),
             -self.f * np.sin(self.f * t),
             self.f * np.cos(self.f * t), 0, 0])[
                                :self.dof]

    @lru_cache(maxsize=None)
    def qdd_des(self, t):
        return self.magnitude * np.array(
            [-self.f ** 2 * np.sin(self.f * t), -self.f ** 2 * np.cos(self.f * t), -self.f ** 2 * np.sin(self.f * t),
             -self.f ** 2 * np.cos(self.f * t), -self.f ** 2 * np.sin(self.f * t), 0, 0])[:self.dof]

    def h(self, e1, e2, t):
        return self.qdd_des(t) - self.alpha @ (self.alpha @ e1) + np.linalg.inv(self.M(t)) @ (
                self.C(t) @ self.qd_des(t)
                # + self.G(t)
                + self.C(t) @ (self.alpha @ e1) - self.C(t) @ e2)

    def q(self, E_t, t):
        e1_t, e2_t = E_t[:self.dof], E_t[self.dof:]
        q = self.q_des(t) - e1_t
        return q

    def dynamic(self, E_t, t, U_delay):
        e1_t, e2_t = E_t[:self.dof], E_t[self.dof:]
        h = self.h(e1_t, e2_t, t)

        e1_t_dot = e2_t - self.alpha @ e1_t
        e2_t_dot = self.alpha @ e2_t + h - np.linalg.inv(self.M(t)) @ U_delay
        return np.concatenate([e1_t_dot, e2_t_dot])

    def kappa(self, E_t, t):
        e1, e2 = E_t[:self.dof], E_t[self.dof:]
        h = self.h(e1, e2, t)
        return self.M(t) @ (h + (self.beta + self.alpha) @ e2)


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

    def __init__(self, c, n, delay=None):
        self.c = c
        self.n = n
        self.delay = delay

    def dynamic(self, Z_t, t, U_delay):
        if len(Z_t.shape) < 2:
            Z1_t, Z2_t = Z_t
            Z1_t_dot = Z2_t - self.c * Z2_t ** self.n * U_delay
            Z2_t_dot = U_delay
            return np.array([Z1_t_dot, Z2_t_dot]).squeeze()
        else:
            U_delay = U_delay.squeeze()
            Z1_t, Z2_t = Z_t[:, 0], Z_t[:, 1]
            Z1_t_dot = Z2_t - self.c * Z2_t ** self.n * U_delay
            Z2_t_dot = U_delay
            return np.vstack([Z1_t_dot, Z2_t_dot]).T

    def kappa(self, Z_t, t):
        Z1 = Z_t[0]
        Z2 = Z_t[1]
        return -Z1 - 2 * Z2 - self.c / (self.n + 1) * Z2 ** (self.n + 1)

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

    def kappa(self, Z_t, t):
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
    def __init__(self, M=1.0, m=0.1, l=1.0, g=9.81, desired_poles=None):
        if desired_poles is None:
            desired_poles = [-2, -2.5, -3, -3.5]
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

    def kappa(self, Z_t, t):
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

    def __init__(self, mu=1.0, desired_poles=None):
        if desired_poles is None:
            desired_poles = [-2, -3]
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

    def kappa(self, Z_t, t):
        u = -np.dot(self.K, Z_t)
        return u

    def dynamic_tensor_batched1(self, Z_t, t, U_delay):
        raise NotImplementedError()

    def dynamic_tensor_batched2(self, Z_t, t, U_delay):
        raise NotImplementedError()


class DynamicSystem3(DynamicSystem):

    @property
    def name(self):
        return 's6'

    @property
    def n_input(self):
        return 2

    @property
    def n_state(self):
        return 3  # dof dimensions for e1 and dof dimensions for e2

    def dynamic(self, X_t, t, U_delay):
        assert X_t.ndim == 1
        X1_t, X2_t, X3_t = X_t
        U1_delay, U2_delay = U_delay
        return np.array([U2_delay * np.cos(X3_t), U2_delay * np.sin(X3_t), U1_delay])

    def kappa(self, X_t, t):
        X1_t, X2_t, X3_t = X_t
        M_t = X1_t * np.cos(X3_t) + X2_t * np.sin(X3_t)
        Q_t = X1_t * np.sin(X3_t) - X2_t * np.cos(X3_t)
        U1 = -M_t ** 2 * np.cos(t) - M_t * Q_t * (1 + np.cos(t) ** 2) - X3_t
        U2 = -M_t + Q_t * (np.sin(t) - np.cos(t)) + Q_t * U1
        return np.array([U1, U2])


class Delay:
    @abstractmethod
    def __call__(self, t):
        ...

    @abstractmethod
    def max_delay(self):
        ...

    @abstractmethod
    def phi(self, t):
        ...

    @abstractmethod
    def phi_prime(self, t):
        ...

    @abstractmethod
    def phi_inverse(self, t):
        ...


class ConstantDelay(Delay):

    def __init__(self, delay: float):
        self.delay = delay

    def __call__(self, t=None):
        return self.delay

    def max_delay(self):
        return self.delay

    def phi(self, t):
        return t - self.delay

    def phi_prime(self, t):
        return 1

    def phi_inverse(self, t):
        return t + self.delay


class TimeVaryingDelay(Delay):

    def __call__(self, t):
        if t < 0:
            t = 0
        return (1 + t) / (1 + 2 * t)

    def max_delay(self):
        return 1

    def phi(self, t):
        if isinstance(t, np.ndarray):
            t[t < 0] = 0
        else:
            t = max(t, 0)
        return t - (1 + t) / (1 + 2 * t)

    def phi_prime(self, t):
        if isinstance(t, np.ndarray):
            t[t < 0] = 0
        else:
            t = max(t, 0)
        return 1 + 1 / (1 + 2 * t) ** 2

    def phi_inverse(self, t):
        # if t < 0:
        #     t = 0
        if isinstance(t, np.ndarray):
            t[t < 0] = 0
            t[t == -1] = 2
            return t + (1 + t) / (((1 + t) ** 2 + 1) ** 0.5 + t)
        else:
            t = max(0, t)
            if t == -1:
                return 2
            return t + (1 + t) / (((1 + t) ** 2 + 1) ** 0.5 + t)


def solve_integral_nn(model, U_D, Z_t, t):
    device = next(model.parameters()).device
    u_tensor = torch.tensor(U_D, dtype=torch.float32, device=device).unsqueeze(0)
    z_tensor = torch.tensor(Z_t, dtype=torch.float32, device=device).unsqueeze(0)
    outputs = model(
        **{
            't': torch.tensor(t),
            'z': z_tensor,
            'u': u_tensor,
            'label': None,
            'input': None
        }
    )
    return outputs.to('cpu').detach().numpy()[0]


def solve_integral(Z_t, P_D, U_D, t: float, dataset_config, delay: Delay):
    assert len(P_D) == len(U_D)
    system = dataset_config.system
    dt = dataset_config.dt

    def f(p, t, u):
        phi_prime = delay.phi_prime(delay.phi_inverse(t))
        dynamic = system.dynamic(p, t, u)
        if isinstance(phi_prime, np.ndarray):
            phi_prime = phi_prime[..., None]
        return dynamic / phi_prime

    n_state = system.n_state
    n_points = len(P_D)
    # assert n_points <= dataset_config.n_point_delay(t)
    ts = np.linspace(t - n_points * dt, t - dt, n_points)
    solution = IntegralSolution()
    if dataset_config.integral_method == 'rectangle':
        solution.solution = solve_integral_rectangle(f=f, Z_t=Z_t, P_D=P_D, U_D=U_D, ts=ts, dt=dt)
    elif dataset_config.integral_method == 'trapezoidal':
        solution.solution = solve_integral_trapezoidal(f=f, Z_t=Z_t, P_D=P_D, U_D=U_D, ts=ts, dt=dt)
    elif dataset_config.integral_method == 'simpson':
        solution.solution = solve_integral_simpson(f=f, Z_t=Z_t, P_D=P_D, U_D=U_D, ts=ts, dt=dt)
    elif dataset_config.integral_method == 'eular':
        solution.solution = solve_integral_eular(Z_t=Z_t, n_points=n_points, n_state=n_state, dt=dt, U_D=U_D, f=f,
                                                 ts=ts)
    elif dataset_config.integral_method == 'successive':
        solution.solution = solve_integral_successive(Z_t=Z_t, n_points=n_points, n_state=n_state, dt=dt, U_D=U_D, f=f,
                                                      n_iterations=dataset_config.successive_approximation_n_iteration,
                                                      adaptive=False, ts=ts)
        solution.n_iter = dataset_config.successive_approximation_n_iteration
    elif dataset_config.integral_method == 'successive adaptive':
        res, n_iter = solve_integral_successive(Z_t=Z_t, n_points=n_points, n_state=n_state, dt=dt, U_D=U_D, f=f,
                                                threshold=dataset_config.successive_approximation_threshold,
                                                adaptive=True, ts=ts)
        solution.solution = res
        solution.n_iter = n_iter
    else:
        raise NotImplementedError()
    return solution


def solve_integral_eular(Z_t, n_points: int, n_state: int, dt: float, U_D: np.ndarray, f, ts):
    if isinstance(Z_t, np.ndarray):
        P_D = np.zeros((n_points + 1, n_state))
    elif isinstance(Z_t, torch.Tensor):
        P_D = torch.zeros((n_points + 1, n_state))
    else:
        raise NotImplementedError()
    P_D[0, :] = Z_t
    for j, t in enumerate(ts):
        P_D[j + 1, :] = P_D[j, :] + dt * f(P_D[j, :], t, U_D[j])
    return P_D[-1]


def solve_integral_successive(Z_t, n_points: int, n_state: int, dt: float, U_D: np.ndarray, f, ts: np.ndarray,
                              n_iterations: int = 1, threshold: float = 1e-5, adaptive: bool = False):
    assert n_iterations >= 0
    assert isinstance(Z_t, np.ndarray)
    if adaptive:
        P_D = np.zeros((2, n_points + 1, n_state))
        P_D[0, :, :] = Z_t
        P_D[1, :, :] = Z_t

        n_iterations = 0
        while True:
            n_iterations += 1
            for j, t in enumerate(ts):
                if j == 0:
                    P_D[1, j + 1, :] = Z_t + dt * f(P_D[0, 0, :], t, U_D[0])
                else:
                    P_D[1, j + 1, :] = P_D[1, j, :] + dt * f(P_D[0, j, :], t, U_D[j])

            # Check for convergence
            if np.all(np.abs(P_D[1] - P_D[0]) < threshold) or n_iterations > 100:
                break

            # Update P_D for the next iteration
            P_D[0, :, :] = P_D[1, :, :]

        return P_D[1, -1, :], n_iterations
    else:
        P_D = np.zeros((n_iterations + 1, n_points + 1, n_state))
        P_D[0, :, :] = Z_t
        P_D[-1, :, :] = Z_t
        for n in range(n_iterations):
            for j, t in enumerate(ts):
                if j == 0:
                    P_D[n + 1, j + 1, :] = Z_t + dt * f(P_D[n, 0, :], t, U_D[0])
                else:
                    P_D[n + 1, j + 1, :] = P_D[n + 1, j, :] + dt * f(P_D[n, j, :], t, U_D[j])
        return P_D[-1, -1, :]


def solve_integral_rectangle(f, Z_t, P_D, U_D, ts, dt: float):
    assert len(P_D) == len(U_D)
    if len(P_D) == 0:
        return Z_t
    integrand_values = np.array([f(p, t, u) for p, t, u in zip(np.array(P_D), np.array(ts), np.array(U_D))])
    integral = integrand_values.sum(0) * dt
    return integral + Z_t


def solve_integral_trapezoidal(f, Z_t, P_D, U_D, ts, dt: float):
    assert len(P_D) == len(U_D)
    if len(P_D) == 0:
        return 0
    integrand_values = f(np.array(P_D), np.array(ts), np.array(U_D))
    t_values = np.arange(0, len(integrand_values)) * dt
    integral = np.trapz(integrand_values, t_values, axis=0)
    return integral + Z_t


def solve_integral_simpson(f, Z_t, P_D, U_D, ts, dt: float):
    assert len(P_D) == len(U_D)
    if len(P_D) == 0:
        return 0
    integrand_values = f(np.array(P_D), np.array(ts), np.array(U_D))
    t_values = np.arange(0, len(integrand_values)) * dt
    integral = simps(integrand_values, t_values, axis=0)
    return integral + Z_t


if __name__ == '__main__':
    ...
