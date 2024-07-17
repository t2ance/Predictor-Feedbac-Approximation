import numpy as np


class BaxterDynamics:
    def __init__(self, link_masses=None, inertia_tensors=None, com_positions=None, dh_parameters=None):
        """
        初始化BaxterDynamics类
        :param link_masses: 连杆质量数组
        :param inertia_tensors: 惯性张量数组
        :param com_positions: 质心位置数组
        :param dh_parameters: Denavit-Hartenberg参数
        """
        if link_masses is None:
            link_masses = [5.70044, 3.22698, 4.31272, 2.07206, 2.24665, 1.60979, 0.54218]

        if inertia_tensors is None:
            inertia_tensors = [
                np.array([[0.0470910226, -0.0061487003, 0.0001278755],
                          [-0.0061487003, 0.035959884, -0.0007808689],
                          [0.0001278755, -0.0007808689, 0.0376697645]]),
                np.array([[0.027885975, -0.0001882199, -0.0008693967],
                          [-0.0001882199, 0.020787492, 0.0020767576],
                          [-0.0008693967, 0.0020767576, 0.0117520941]]),
                np.array([[0.0266173355, -0.0039218988, 0.0002927063],
                          [-0.0039218988, 0.0124803073, -0.001083893],
                          [0.0002927063, -0.001083893, 0.0284435520]]),
                np.array([[0.0131822787, -0.0001966341, 0.0003603617],
                          [-0.0001966341, 0.00926852, 0.000745949],
                          [0.0003603617, 0.000745949, 0.0071158268]]),
                np.array([[0.0166748282, -0.0001865762, 0.0001840370],
                          [-0.0001865762, 0.003746311, 0.0004673235],
                          [0.0001840370, 0.0004673235, 0.0167545726]]),
                np.array([[0.0070053791, 0.0001534806, -0.0004438478],
                          [0.0001534806, 0.005527552, -0.0002111503],
                          [-0.0004438478, -0.0002111503, 0.0038760715]]),
                np.array([[0.0008162135, 0.000128440, 0.00018969891],
                          [0.000128440, 0.0008735012, 0.0001057726],
                          [0.00018969891, 0.0001057726, 0.0005494148]])
            ]

        if com_positions is None:
            com_positions = [
                np.array([-0.05117, 0.07908, 0.00086]),
                np.array([0.00269, -0.00529, 0.06845]),
                np.array([-0.07176, 0.08149, 0.00132]),
                np.array([0.00159, -0.01117, 0.02618]),
                np.array([-0.01168, 0.13111, 0.0046]),
                np.array([0.00697, 0.006, 0.06048]),
                np.array([0.005137, 0.0009572, -0.06682])
            ]

        if dh_parameters is None:
            dh_parameters = [
                (0, 0.2703, 0.069, -np.pi / 2),
                (0, 0, 0, np.pi / 2),
                (0, 0.3644, 0.069, -np.pi / 2),
                (0, 0, 0, np.pi / 2),
                (0, 0.3743, 0.01, -np.pi / 2),
                (0, 0, 0, np.pi / 2),
                (0, 0.2295, 0, 0)
            ]

        self.link_masses = link_masses
        self.inertia_tensors = inertia_tensors
        self.com_positions = com_positions
        self.dh_parameters = dh_parameters
        self.num_links = len(link_masses)

        # 定义重力矢量
        self.gravity = np.array([0, 0, -9.81, 0])

    def get_transform_matrix(self, theta, d, a, alpha):
        """
        计算齐次变换矩阵
        :param theta: 旋转角度
        :param d: 沿Z轴的偏移
        :param a: 沿X轴的偏移
        :param alpha: 扭转角
        :return: 齐次变换矩阵
        """
        return np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def get_jacobian_matrix(self, inertia_tensor, mass, com_position):
        """
        计算Jacobian矩阵
        :param inertia_tensor: 惯性张量
        :param mass: 连杆质量
        :param com_position: 质心位置
        :return: Jacobian矩阵
        """
        Ixx, Iyy, Izz = inertia_tensor[0, 0], inertia_tensor[1, 1], inertia_tensor[2, 2]
        Ixy, Ixz, Iyz = inertia_tensor[0, 1], inertia_tensor[0, 2], inertia_tensor[1, 2]
        x, y, z = com_position
        return np.array([
            [(Ixx + Iyy - Izz) / 2, Ixy, Ixz, mass * x],
            [Ixy, (Iyy + Izz - Ixx) / 2, Iyz, mass * y],
            [Ixz, Iyz, (Izz + Ixx - Iyy) / 2, mass * z],
            [mass * x, mass * y, mass * z, mass]
        ])

    def compute_inertia_matrix(self, q):
        """
        计算惯性矩阵M(q)
        :param q: 关节角度数组
        :return: 惯性矩阵M
        """
        M = np.zeros((self.num_links, self.num_links))
        for i in range(self.num_links):
            for j in range(i, self.num_links):
                T = np.eye(4)
                for k in range(i, j + 1):
                    theta, d, a, alpha = self.dh_parameters[k]
                    T = T @ self.get_transform_matrix(theta + q[k], d, a, alpha)
                J = self.get_jacobian_matrix(self.inertia_tensors[j], self.link_masses[j], self.com_positions[j])
                M[i, j] = np.trace(T[:3, :3] @ J[:3, :3])
                if i != j:
                    M[j, i] = M[i, j]
        return M

    def compute_coriolis_centrifugal_matrix(self, q, q_dot):
        """
        计算科氏/离心力矩阵C(q, q_dot)
        :param q: 关节角度数组
        :param q_dot: 关节角速度数组
        :return: 科氏/离心力矩阵C
        """
        C = np.zeros((self.num_links, self.num_links))
        for i in range(self.num_links):
            for j in range(self.num_links):
                for k in range(self.num_links):
                    T = np.eye(4)
                    for l in range(max(i, j, k)):
                        theta, d, a, alpha = self.dh_parameters[l]
                        T = T @ self.get_transform_matrix(theta + q[l], d, a, alpha)
                    J = self.get_jacobian_matrix(self.inertia_tensors[j], self.link_masses[j], self.com_positions[j])
                    C[i, j] += np.trace(T[:3, :3] @ J[:3, :3]) * q_dot[k]
        return C

    def compute_gravity_vector(self, q):
        """
        计算重力向量G(q)
        :param q: 关节角度数组
        :return: 重力向量G
        """
        G = np.zeros(self.num_links)
        for i in range(self.num_links):
            T = np.eye(4)
            for j in range(i + 1):
                theta, d, a, alpha = self.dh_parameters[j]
                T = T @ self.get_transform_matrix(theta + q[j], d, a, alpha)
            G[i] = -self.link_masses[i] * np.dot(self.gravity[:3], T[:3, 3])
        return G

    def compute_dynamics(self, t, y, tau):
        """
        计算动力学方程的右侧，即状态导数
        :param t: 时间
        :param y: 状态向量 [q, q_dot]
        :param tau: 外部施加的关节扭矩
        :return: 状态导数 [q_dot, q_ddot]
        """
        q = y[:self.num_links]
        q_dot = y[self.num_links:]

        M = self.compute_inertia_matrix(q)
        C = self.compute_coriolis_centrifugal_matrix(q, q_dot)
        G = self.compute_gravity_vector(q)

        q_ddot = np.linalg.inv(M).dot(tau - C.dot(q_dot) - G)

        return np.concatenate((q_dot, q_ddot))


if __name__ == '__main__':
    from scipy.integrate import solve_ivp

    baxter_dynamics = BaxterDynamics()

    # 定义初始状态 [q0, q_dot0]
    q0 = np.zeros(7)
    q_dot0 = np.zeros(7)
    y0 = np.concatenate((q0, q_dot0))

    # 定义仿真时间
    t_span = [0, 10]  # 仿真10秒
    tau = np.zeros(7)  # 假设关节扭矩为零

    # 使用solve_ivp求解ODE
    # solution = solve_ivp(baxter_dynamics.compute_dynamics, t_span, y0, args=(tau,), t_eval=np.linspace(0, 10, 500))
    # solution = solve_ivp(baxter_dynamics.compute_dynamics, t_span, y0, args=(tau,), t_eval=np.linspace(0, 10, 100),
    #                      max_step=0.01)
    solution = solve_ivp(baxter_dynamics.compute_dynamics, t_span, y0, args=(tau,), t_eval=np.linspace(0, 10, 100),
                         rtol=1e-5, atol=1e-8)

    # 输出仿真结果
    import matplotlib.pyplot as plt

    plt.plot(solution.t, solution.y[:7].T)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Angles [rad]')
    plt.title('Baxter Robot Joint Angles Over Time')
    plt.legend(['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7'])
    plt.show()
