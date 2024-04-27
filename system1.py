import numpy as np


def system(Z_t, t, U_t_minus_D):
    Z1_t = Z_t[0]
    Z2_t = Z_t[1]
    Z1_t_dot = Z2_t - Z2_t ** 2 * U_t_minus_D
    Z2_t_dot = U_t_minus_D
    return np.array([Z1_t_dot, Z2_t_dot])


def control_law_explict(t, Z0, D):
    z1_0, z2_0 = Z0
    if t >= 0:
        term1 = z1_0 + (2 + D) * z2_0 + (1 / 3) * z2_0 ** 3
        term2 = z1_0 + (1 + D) * z2_0 + (1 / 3) * z2_0 ** 3
        u_t = -np.exp(D - t) * (term1 + (D - t) * term2)
        return u_t
    elif t >= -D:
        return 0
    else:
        raise NotImplementedError()


def control_law(P_t):
    p1t = P_t[0]
    p2t = P_t[1]
    return -p1t - 2 * p2t - 1 / 3 * p2t ** 3


def predict_integral_general(f, Z_t, P_D, U_D, dt, t):
    assert len(P_D) == len(U_D)
    integral = sum([f(p, t, u) for p, u in zip(P_D, U_D)]) * dt
    return integral + Z_t


def ode_forward(Z_t, Z_t_dot, dt):
    return Z_t + Z_t_dot * dt


def solve_z_explict(t, D, Z0):
    if t < 0:
        raise NotImplementedError()
    if t < D:
        return Z0[0] + Z0[1] * t, Z0[1]
    z1_0 = Z0[0]
    z2_0 = Z0[1]
    z2_D = Z0[1]
    middle_term = z1_0 + D * z2_0 + (1 / 3) * z2_0 ** 3
    term1 = np.exp(D - t) * ((1 + t - D) * middle_term + (t - D) * z2_D)
    term2 = - (1 / 3) * np.exp(3 * (D - t)) * ((D - t) * middle_term + (1 - t + D) * z2_D) ** 3
    Z1 = term1 + term2

    Z2 = np.exp(D - t) * ((D - t) * middle_term + (1 - t + D) * z2_D)
    return Z1, Z2


def predict_integral_explict(t, delay, Z1_t, Z2_t, U_D, ts_D, dt):
    assert len(ts_D) == len(U_D)
    term1 = sum(U_D) * dt
    term2 = sum([(t - theta) * u * dt for u, theta in zip(U_D, ts_D)])
    P1_t = Z1_t + delay * Z2_t + term2 - Z2_t ** 2 * term1 - Z2_t * term1 ** 2
    P2_t = Z2_t + term1
    return P1_t, P2_t


if __name__ == '__main__':
    ...
