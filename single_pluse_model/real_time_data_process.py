import numpy as np
from . import RD_single_pulse
from . import model
import torch
import os
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import pandas as pd

def calculate_orbital_elements(miu, R0, V0):
    """
    根据位置和速度矢量求出相应的六根数
    Args:
        miu: 3.986E5  # km3/s2
             3.986E14  # m3/s2
        R0: 位置
        V0: 速度

    Returns:
        data: 椭圆和双曲线转移轨道的六根数: a;e;i;omega;Omega;f
        %a: 半长轴；e：离心率；i：轨道倾角；
        %omega: 近地点幅角；Omega: 升交点经度；f: 真近点角。
        抛物线转移轨道的根数: p;i;omega;Omega;f
        %p：半矩形矩；i：轨道倾角；omega：近地点幅角；
        %Omega: 升交点经度；f: 真近点角。
        圆转移轨道的根数: a;i;u;Omega
        %a：半径；i：轨道倾角；u：纬度参数；
        %Omega: 升交点经度。
    """

    r_norm = np.linalg.norm(R0)
    v_norm = np.linalg.norm(V0)
    r_dot_v = np.dot(R0, V0)
    if 2 / r_norm - v_norm ** 2 / miu != 0:
        # 计算椭圆轨道的半长轴
        a = 1 / abs(2 / r_norm - v_norm ** 2 / miu)
    else:
        a = None

    # 计算轨道离心率
    E = (v_norm ** 2 / miu - 1 / r_norm) * R0 - r_dot_v / miu * V0
    e = np.linalg.norm(E)

    # 计算角动量矢量
    H = np.cross(R0, V0)
    # 计算角动量的模长
    h = np.linalg.norm(H)
    # 计算抛物线轨道的半矩形矩
    p = h ** 2 / miu

    Z = np.array([0, 0, 1])
    X = np.array([1, 0, 0])
    Y = np.array([0, 1, 0])
    N = np.cross(Z, H)
    # 计算升交点赤经的单位矢量
    n = np.linalg.norm(N)
    # 计算轨道倾角
    i = np.arccos(np.dot(Z, H) / h)

    if e != 0:
        # 计算近地点幅角
        if n != 0 and e != 0:
            omega = np.arccos(np.dot(N, E) / n / e)
        else:
            omega = 0.0
        # omega = np.arccos(np.dot(N, E) / n / e)
        # if np.isnan(omega):
        #     omega = 0.0  # 如果计算结果为 nan，将值设为 0
        if np.dot(Z, E) < 0:
            omega = 2 * np.pi - omega
    else:
        # 计算纬度参数
        u = np.arccos(np.dot(N, R0) / n / r_norm)
        if np.dot(R0, Z) < 0:
            u = 2 * np.pi - u

    # 计算升交点赤经
    if n != 0:
        Omega = np.arccos(np.dot(X, N) / n)
    else:
        Omega = 0.0
    # Omega = np.arccos(np.dot(X, N) / n)
    # if np.isnan(Omega):
    #     Omega = 0.0  # 如果计算结果为 nan，将值设为 0
    if np.dot(Y, N) < 0:
        Omega = 2 * np.pi - Omega

    if e != 0:
        # 计算真近点角
        f = np.arccos(np.dot(E, R0) / e / r_norm)
        if r_dot_v < 0:
            f = 2 * np.pi - f

    # 如果速度矢量的模长与两倍位置矢量的模长之差不为零
    if 2 / r_norm - v_norm ** 2 / miu != 0:
        if e != 0:
            data = [a, e, i, omega, Omega, f]
        else:
            data = [a, i, u, Omega]
    else:
        data = [p, i, omega, Omega, f]

    return data

# 使用数值法求解椭圆拟合参数
def numerical_method_process(R0, V0, fuel):
    input = calculate_orbital_elements(3.986e14, R0, V0)
    ellipse_params = RD_single_pulse.Incoming_parameters(input, fuel)
    return ellipse_params

def network_method_process(R0, V0, fuel):
    input = []
    orbit_data = calculate_orbital_elements(3.986e14, R0, V0)
    input = orbit_data[:3] + orbit_data[5:]
    input.append(fuel)
    input = torch.tensor(input, dtype=torch.float32)

    net = model.ImprovedNN()

    model_name = '/mnt/datab/home/yuanwenzheng/PycharmProjects/ppo_flight/single_pluse_model'
    save_path = os.path.join(model_name, 'MLPNet.pth')
    net.load_state_dict(torch.load(save_path))
    predicted_output = net(input)
    return predicted_output

class network_method_train():
    def __init__(self, pretrain = False):
        self.net = model.ImprovedNN()
        if pretrain:
            now_path = '/mnt/datab/home/yuanwenzheng/PycharmProjects/ppo_flight/single_pluse_model'
            save_path = os.path.join(now_path, 'MLPNet.pth')
            self.net.load_state_dict(torch.load(save_path))
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.count = 0
        self.loss = []
        self.all_loss = []


        # 数据标准化
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

    def train(self, R0, V0, fuel, target):
        target = self.output_scaler.fit_transform(target.reshape(1,-1))
        target = target.reshape(10)
        target = torch.tensor(target, dtype=torch.float32)
        input = []
        orbit_data = calculate_orbital_elements(3.986e14, R0, V0)
        input = orbit_data[:3] + orbit_data[5:]
        input.append(fuel)
        input = np.array(input)
        input = self.input_scaler.fit_transform(input.reshape(1, -1))
        input = input.reshape(5)
        input = torch.tensor(input, dtype=torch.float32)

        self.optimizer.zero_grad()
        predictions = self.net(input)
        loss = self.criterion(predictions, target)
        self.loss.append(loss)
        self.all_loss.append(loss)
        loss.backward()
        self.optimizer.step()


        self.count += 1
        if self.count == 10:
            self.scheduler.step()
            self.count = 0
            print("loss: ", sum(self.loss) / 10)
            self.loss = []
            model_name = 'MLPNet2' + '.pth'
            now_path = '/mnt/datab/home/yuanwenzheng/PycharmProjects/ppo_flight/single_pluse_model'
            chkpt_file = os.path.join(now_path, model_name)
            torch.save(self.net.state_dict(), chkpt_file)
            # predicted_output = self.output_scaler.inverse_transform(predictions.detach().numpy().reshape(1, -1))
            # actual_output = self.output_scaler.inverse_transform(target.unsqueeze(0).detach().numpy().reshape(1, -1))

        if len(self.all_loss) % 200 == 0:
            all_loss_np = np.array([l.detach().numpy() for l in self.all_loss])
            df = pd.DataFrame(all_loss_np, columns=['loss'])
            output_file = 'all_loss.xlsx'
            df.to_excel(output_file, index=False)
            print(f"loss数据已成功保存到 {output_file}")


