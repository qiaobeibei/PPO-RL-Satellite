import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""https://blog.csdn.net/weixin_57997461/article/details/137403155"""

plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False
mu = 398600
Re = 6378.137  # 地球半径
J2 = 0.00108263  # J2项


##状态方程
def StateEq(t, RV):
    x = RV[0]
    y = RV[1]
    z = RV[2]
    vx = RV[3]
    vy = RV[4]
    vz = RV[5]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    gx = -mu * x / r ** 3
    gy = -mu * y / r ** 3
    gz = -mu * z / r ** 3
    dgx = -3 / 2 * J2 * Re ** 2 * mu * x / r ** 5 * (1 - 5 * (z / r) ** 2)
    dgy = -3 / 2 * J2 * Re ** 2 * mu * y / r ** 5 * (1 - 5 * (z / r) ** 2)
    dgz = -3 / 2 * J2 * Re ** 2 * mu * z / r ** 5 * (3 - 5 * (z / r) ** 2)
    f = np.array([vx, vy, vz, gx + dgx, gy + dgy, gz + dgz])
    return f


## 龙格库塔算法
def RungeKutta(t0, r0, h):
    K1 = StateEq(t0, r0)
    K2 = StateEq(t0 + 2 / h, r0 + h / 2 * K1)
    K3 = StateEq(t0 + h, r0 + h / 2 * K2)
    K4 = StateEq(t0 + h, r0 + h * K3)
    r1 = r0 + h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
    return r1


h = 1  # 步长
N = 86400  # 外推时间，一天86400秒
# 初始位置和速度
RV0 = np.array([3971.676026, -2202.172866, -5161.178823, 6.059801, 3.231769, 3.293050])
data = np.array(np.zeros((np.int32(N / h + 1), 6)))
data[0] = RV0
for i in range(np.int32(N / h)):
    data[i + 1] = RungeKutta(i * h, RV0, h)
    RV0 = data[i + 1]
print(RV0)
##
df = pd.read_csv(r"J2_J2000_Position_Velocity.csv")
error = df.iloc[:, 1:] - data
plt.subplot(2, 2, 1)
plt.plot(error.iloc[:, 0:3], label=['Xerror', 'Yerror', 'Zerror'])
plt.xlabel('时间/s')
plt.ylabel('位置误差/km')
plt.legend()
plt.grid()
plt.title('J2数值外推与STK的J2模型外推的位置误差')
plt.subplot(2, 2, 2)
plt.plot(error.iloc[:, 3:6], label=['VXerror', 'VYerror', 'VZerror'])
plt.xlabel('时间/s')
plt.ylabel('速度误差/(km/s)')
plt.legend()
plt.grid()
plt.title('J2数值外推与STK的J2模型外推的速度误差')

df = pd.read_csv(r"HPOP_J2000_Position_Velocity.csv")
error = df.iloc[:, 1:] - data
plt.subplot(2, 2, 3)
plt.plot(error.iloc[:, 0:3], label=['Xerror', 'Yerror', 'Zerror'])
plt.xlabel('时间/s')
plt.ylabel('位置误差/km')
plt.legend()
plt.grid()
plt.title('J2数值外推与STK的HPOP模型外推的位置误差')
plt.subplot(2, 2, 4)
plt.plot(error.iloc[:, 3:6], label=['VXerror', 'VYerror', 'VZerror'])
plt.xlabel('时间/s')
plt.ylabel('速度误差/(km/s)')
plt.legend()
plt.grid()
plt.title('J2数值外推与STK的HPOP模型外推的速度误差')