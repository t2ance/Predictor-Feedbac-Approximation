import numpy as np

from config import DatasetConfig
from main import run


def compute_Z_at_t_plus_D(U_vector, Z_at_t, delta_t):
    def f(Z, U_delayed):
        dZ1_dt = Z[1] - Z[0] ** 2 * U_delayed
        dZ2_dt = U_delayed
        return np.array([dZ1_dt, dZ2_dt])

    Z_current = np.copy(Z_at_t)
    num_steps = len(U_vector)

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


if __name__ == '__main__':
    dataset_config = DatasetConfig(noise_sigma_numerical=0.1)
    U, Z, P = run(method='numerical', silence=True, Z0=(0., 1.), plot=True, dataset_config=dataset_config)
    # U, Z, P = run(method='numerical', silence=True, Z0=(3., 3.), plot=True, dataset_config=dataset_config)
    # U, Z, P = run(method='numerical', silence=True, Z0=(-3., -3.), plot=True, dataset_config=dataset_config)
    # U, Z, _ = run(method='explict', silence=True, duration=dataset_config.duration, delay=dataset_config.delay,
    #               Z0=(0, 1), dt=dataset_config.dt, plot=True)
