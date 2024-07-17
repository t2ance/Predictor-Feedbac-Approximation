import numpy as np


class BaxterParameters:
    def __init__(self, link_masses=None, inertia_tensors=None, com_positions=None, dh_parameters=None):
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

        self.gravity = np.array([0, 0, -9.81, 0])

    def get_transform_matrix(self, theta, d, a, alpha):
        return np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def compute_Uij(self, q):
        num_links = self.num_links
        Uij = np.zeros((num_links, num_links, 4, 4))
        Qj = np.array([
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        for i in range(num_links):
            for j in range(num_links):
                if j <= i:
                    T = np.eye(4)
                    for k in range(j):
                        theta, d, a, alpha = self.dh_parameters[k]
                        T = T @ self.get_transform_matrix(theta + q[k], d, a, alpha)
                    Uij[i, j] = T @ Qj @ np.linalg.inv(T)
        return Uij

    def compute_Uijk(self, Uij, q):
        num_links = self.num_links
        Uijk = np.zeros((num_links, num_links, num_links, 4, 4))

        for i in range(num_links):
            for j in range(num_links):
                for k in range(num_links):
                    if i >= k >= j:
                        T = np.eye(4)
                        for l in range(k):
                            theta, d, a, alpha = self.dh_parameters[l]
                            T = T @ self.get_transform_matrix(theta + q[l], d, a, alpha)
                        Qj = Uij[j, i]
                        Qk = Uij[k, i]
                        Uijk[i, j, k] = T @ Qk @ Qj
        return Uijk

    def compute_Ji(self, inertia_tensor, mass, com_position):
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
        M = np.zeros((self.num_links, self.num_links))
        Uij = self.compute_Uij(q)

        for i in range(self.num_links):
            for k in range(self.num_links):
                sum_trace = 0
                for j in range(max(i, k), self.num_links):
                    Ji = self.compute_Ji(self.inertia_tensors[j], self.link_masses[j], self.com_positions[j])
                    sum_trace += np.trace(Uij[j, k] @ Ji @ Uij[j, i].T)
                M[i, k] = sum_trace
        return M

    def compute_coriolis_centrifugal_matrix(self, q, q_dot):
        num_links = self.num_links
        C = np.zeros(num_links)
        Uij = self.compute_Uij(q)
        Uijk = self.compute_Uijk(Uij, q)

        for i in range(num_links):
            for k in range(num_links):
                for m in range(num_links):
                    hikm = 0
                    for j in range(max(i, k, m), num_links):
                        Ji = self.compute_Ji(self.inertia_tensors[j], self.link_masses[j], self.com_positions[j])
                        hikm += np.trace(Uijk[j, k, m] @ Ji @ Uij[j, i].T)
                    C[i] += hikm * q_dot[k] * q_dot[m]

        return C

    def compute_gravity_vector(self, q):
        num_links = self.num_links
        G = np.zeros(num_links)
        Uij = self.compute_Uij(q)

        for i in range(num_links):
            Gi = 0
            for j in range(i, num_links):
                rj = np.append(self.com_positions[j], 1)
                Gi += -self.link_masses[j] * np.dot(self.gravity, (Uij[j, i] @ rj))
            G[i] = Gi

        return G


if __name__ == '__main__':
    ...
