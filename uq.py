import math
from scipy.stats import norm
from sortedcontainers import SortedList


class GaussianQuantileEstimator:
    def __init__(self):
        """
        初始化高斯分布的估计器，维护样本均值和方差。
        """
        self.n = 0  # 样本数量
        self.mean = 0  # 样本均值
        self.M2 = 0  # 样本的平方和，用于计算方差

    def update(self, x):
        """
        接受一个新的数据点 x，并更新均值和方差。
        使用增量更新公式计算样本均值和方差。
        """
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def variance(self):
        """
        返回样本方差。若样本数不足2个，返回0.
        """
        if self.n < 2:
            return 0
        return self.M2 / (self.n - 1)

    def query(self, q):
        """
        查询给定分位数 q 对应的值，基于高斯分布。
        q 应为 [0, 1] 之间的浮点数。
        """
        if not 0 <= q <= 1:
            raise ValueError("q should be between 0 and 1")

        if self.n == 0:
            raise ValueError("No data has been provided yet.")

        std_dev = math.sqrt(self.variance())

        # 使用标准正态分布的分位数函数 ppf 计算
        z_score = norm.ppf(q)  # 标准正态分布的分位数
        return self.mean + z_score * std_dev


class QuantileEstimator:
    def __init__(self):
        """
        初始化分位数估计器
        """
        self.data = SortedList()  # 用于存储有序的数据

    def update(self, x):
        """
        接收新的数据点 x，并将其插入到有序列表中。
        """
        self.data.add(x)

    def query(self, q):
        """
        查询 q 分位数的值，q 应为 [0, 1] 之间的浮点数
        """
        if not 0 <= q <= 1:
            raise ValueError("q should be between 0 and 1")

        # 根据 q 找到对应的索引位置
        pos = int(q * (len(self.data) - 1))  # 注意：索引是从 0 开始的
        return self.data[pos]


if __name__ == '__main__':
    ...
