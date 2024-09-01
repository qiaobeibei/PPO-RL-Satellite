import numpy as np
import gym
from gym import spaces
# 这个类用于动态计算数据的均值和标准差。在初始化时，它接收一个参数shape，表示输入数据的维度。
# 它具有三个属性：n（数据点的数量）、mean（均值）和S（用于计算标准差的中间变量）。
# 在每次更新数据时，它会根据新的数据点更新均值和标准差。
class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0       # 下面声明其基本属性
        M = 0.4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        self.action_space = [-M, -M / 2, 0, M / 2, M, -M, -M / 2, 0, M / 2, M, -M, -M / 2, 0, M / 2, M,-M, -M / 2, 0, M / 2, M]  # 定义动作空间中的具体动作值

        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

# 这个类用于对输入数据进行归一化。它包含一个RunningMeanStd对象，用于计算均值和标准差。
# 在__call__方法中，它接收输入数据x，并根据是否需要更新均值和标准差来对数据进行归一化处理。
class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


# 这个类用于调整奖励信号的尺度。它包含一个RunningMeanStd对象，用于计算奖励信号的均值和标准差。
# 在__call__方法中，它接收奖励信号x，然后按照一定的尺度进行调整。
# 在每个时间步上，它将上一个时间步的奖励信号乘以折扣因子gamma，然后将其累加到当前的奖励总和self.R中，并根据累计奖励信号的均值和标准差对当前奖励信号进行归一化。
class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)

