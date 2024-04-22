import numpy as np
from scipy.integrate import solve_ivp

from config import DatasetConfig
from main import create_trajectory_dataset, f, predict_by_integral, run


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


if __name__ == '__main__':
    # samples_ = create_trajectory_dataset(
    #     1, dataset_config.duration, dataset_config.delay, 800, dataset_config.n_delay_step,
    #     dataset_config.n_sample_per_dataset, dataset_config.dataset_file)
    # loss = 0.
    # for sample in samples_:
    #     features, label = sample
    #     t = features[:1].cpu().numpy()
    #     z = features[1:3].cpu().numpy()
    #     u = features[3:].cpu().numpy()
    #     label = label.cpu().numpy()
    #     P = compute_Z_at_t_plus_D(u, z, dataset_config.dt)
    #     loss += (P - label) ** 2
    # P = compute_Z_at_t_plus_D(u, z, dataset_config.dt)
    # print(loss)
    # Constants
    # U_vector_example = np.random.rand(300)  # Assuming D=3 and delta_t=0.01, so 300 steps
    # Z_at_t0_example = np.array([0, 1])  # Example Z(t0) values
    #
    # # Compute P(t0) without knowing the specific time t0
    # P_t0_example = compute_Z_at_t_plus_D(U_vector_example, Z_at_t0_example, 0.01)
    # P_t0_example
    duration = 8
    n_point = 800
    second_th = 3
    delay = 3
    dt = duration / n_point
    idx = int(n_point * (second_th / duration))
    n_delay_step = int(n_point * (delay / duration))
    U, Z, P = run(
        method='explict', silence=True, duration=duration, D=delay, Z0=(0, 1), n_point=n_point, plot=True
    )
    # U, Z, P = run(
    #     method='numerical', silence=True, duration=duration, D=delay, Z0=(0, 1),
    #     n_point=n_point, plot=True, plot_predict=True
    # )
    # U_input = U[idx - n_delay_step:idx]
    # Z_input = Z[idx: idx + n_delay_step]
    # assert len(U_input) == len(Z_input)
    # Z_t = Z[idx]
    # Z_t_plus_D = Z[idx + n_delay_step]
    # integral = np.array([f(z, u) for z, u in zip(Z_input, U_input)]).sum(axis=0) * dt
    # Z_t_plus_D_predict = Z_t + integral
    # z_t_D_2 = np.sum(U_input) * dt + Z_t[1]
