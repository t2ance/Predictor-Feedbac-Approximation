import numpy as np


class DynamicSystem:
    '''
    The dynamic system with constant C
    '''

    def __init__(self, delay, Z0, c):
        self.delay = delay
        self.Z0 = Z0
        self.c = c

    def dynamic(self, Z_t, t, U_delay):
        Z1_t = Z_t[0]
        Z2_t = Z_t[1]
        Z1_t_dot = Z2_t - self.c * Z2_t ** 2 * U_delay
        Z2_t_dot = U_delay
        return np.array([Z1_t_dot, Z2_t_dot])

    def kappa(self, Z_t):
        Z1 = Z_t[0]
        Z2 = Z_t[1]
        return -Z1 - 2 * Z2 - self.c / 3 * Z2 ** 3

    def U(self, t):
        assert self.c == 1
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

    def Z(self, t):
        assert self.c == 1
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


def predict_integral(Z_t, n_point_delay: int, n_state: int, dt: float, U_D: np.ndarray, system):
    P_D = np.zeros((n_point_delay, n_state))
    P_D[0, :] = Z_t
    for j in range(n_point_delay - 1):
        P_D[j + 1, :] = P_D[j, :] + dt * system(P_D[j, :], j * dt, U_D[j])
    p = predict_integral_general(f=system, Z_t=Z_t, P_D=P_D, U_D=U_D, dt=dt, t=dt * n_point_delay)
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
