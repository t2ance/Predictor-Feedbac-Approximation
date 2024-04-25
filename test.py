import numpy as np
from matplotlib import pyplot as plt

from config import DatasetConfig
from main import run, integral_prediction_explict, control_law, system


def compute_Z_at_t_plus_D(U_vector, Z_at_t, delta_t):
    # Define the function f for the system, which is assumed to be same as before
    def f(Z, U_delayed):
        dZ1_dt = Z[1] - Z[0] ** 2 * U_delayed
        dZ2_dt = U_delayed
        return np.array([dZ1_dt, dZ2_dt])

    # Initialize variables
    Z_current = np.copy(Z_at_t)  # Start with the initial condition Z(t)
    num_steps = len(U_vector)  # Number of steps to integrate over

    # Time integration using the trapezoidal rule
    for i in range(num_steps - 1):
        # Current and next value of U, considering the delay
        U_current = U_vector[i]
        U_next = U_vector[i + 1]

        # Estimate the derivative at the current and next time step
        f_current = f(Z_current, U_current)
        f_next = f(Z_current + f_current * delta_t, U_next)  # Euler step to estimate next derivative

        # Trapezoidal rule to update Z
        Z_next = Z_current + (delta_t / 2) * (f_current + f_next)

        # Prepare for next iteration
        Z_current = Z_next

    # At the end of the loop, Z_current is Z(t+D)
    return Z_current


def simulation():
    duration = 8
    D = 3
    dt = 0.001
    n_point = int(duration / dt)
    n_point_delay = int(D / dt)
    n_point_total = n_point + n_point_delay
    ts = np.linspace(-D, duration, n_point_total)

    U = np.zeros(n_point_total)
    Z = np.zeros((n_point_total, 2))
    P = np.zeros((n_point_total, 2))
    Z[n_point_delay, :] = [0, 1]

    for i in range(n_point_delay, n_point_total):
        t_i = i
        t_minus_D_i = i - n_point_delay
        Z1_t = Z[t_i, 0]
        Z2_t = Z[t_i, 1]
        P1_t, P2_t = integral_prediction_explict(t=ts[t_i], delay=D, Z1_t=Z1_t, Z2_t=Z2_t, U_D=U[t_minus_D_i:t_i],
                                                 ts_D=ts[t_minus_D_i:t_i], dt=dt)
        # P1_t, P2_t = integral_prediction_general(f=system, Z_t=Z[t_i, :], P_D=P[t_minus_D_i:t_i],
        #                                          U_D=U[t_minus_D_i:t_i], dt=dt)
        P[t_i, :] = [P1_t, P2_t]
        U_t = control_law(P[t_i, :])
        Z1_t_dot, Z2_t_dot = system(Z[t_i, :], U[t_minus_D_i])
        Z1_t_plus_1 = Z1_t + Z1_t_dot * dt
        Z2_t_plus_1 = Z2_t + Z2_t_dot * dt
        if t_i + 1 < n_point_total:
            Z[t_i + 1, :] = [Z1_t_plus_1, Z2_t_plus_1]
        U[t_i] = U_t

    P = P[n_point_delay:, :]
    U = U[n_point_delay:]
    Z = Z[n_point_delay:, :]
    ts = ts[n_point_delay:]

    plt.plot(ts, Z[:, 0], label='$Z_1(t)$')
    plt.xlabel('time')
    plt.ylabel('$Z_1(t)$')
    plt.grid(True)
    plt.show()

    plt.plot(ts, Z[:, 1], label='$Z_2(t)$')
    plt.xlabel('time')
    plt.ylabel('$Z_2(t)$')
    plt.grid(True)
    plt.show()

    plt.plot(ts, U, label='$U(t)$', color='black')
    plt.xlabel('time')
    plt.ylabel('$U(t)$')
    plt.grid(True)
    plt.show()

    plt.plot(ts, P[:, 0], label='$P_1(t)$')
    plt.ylabel('$P_1(t)$')
    plt.grid(True)
    plt.show()

    plt.plot(ts, P[:, 1], label='$P_2(t)$')
    plt.ylabel('$P_2(t)$')
    plt.grid(True)
    plt.show()

    for idx in range(2):
        plt.plot(ts[n_point_delay:], P[:-n_point_delay, idx], label=f'$P_{idx + 1}(t-{D})$')
        plt.plot(ts[n_point_delay:], Z[n_point_delay:, idx], label=f'$Z_{idx + 1}(t)$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dataset_config = DatasetConfig()
    U, Z, P = run(method='numerical', silence=True, duration=dataset_config.duration, delay=dataset_config.delay,
                  Z0=(0, 1), dt=dataset_config.dt, plot=True)
