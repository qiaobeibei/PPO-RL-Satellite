import numpy as np
from tqdm import tqdm
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from . import curve_fitting as cf

params = {
    'a': 10**7,                # 半长轴
    'i': 0,
    'e0': 0.2,
    'f': np.pi / 2,
    'delta_max': 500,
    'u': 3.986e14,        # 地球引力常数
    'N1': 1,
    'N2': 200,
    'N3': 200,
    'delta_l': 1500
}

def Incoming_parameters(data, delta_max):
    """
    接受航天器的轨道参数并保存在全局字典中
    :param data: 航天器轨道六根数
    :return: params
    """
    global params
    params['a'] = data[0]
    params['i'] = data[2]
    params['e0'] = data[1]
    params['f'] = data[5]
    params['delta_max'] = delta_max

    RF_combined = Reachable_Domain()

    return np.array(RF_combined)


def Reachable_Domain():
    a = params['a']                # 半长轴
    i = params['i']                     # 轨道倾角
    e0 = params['e0']                   # 偏心率
    f = params['f']          # 机动位置(真近点角)
    delta_max = params['delta_max']            # 最大速度脉冲 ▲v
    u = params['u']        # 引力常数
    r0 = a * (1 - e0**2) / (1 + e0 * np.cos(f))     # 位置矢量
    p0 = a * (1 - e0**2)       # 轨道半通径

    v1 = np.sqrt(u * (2 / r0 - 1 / a))     # 速度大小
    R0 = np.array([r0 * np.cos(f), r0 * np.sin(f), 0])      # 初始位置矢量

    N1 = params['N1']
    N2 = params['N2']
    N3 = params['N3']
    delta_l = params['delta_l']
    RF_max = []         # 矢径极大值
    RF_min = []         # 矢径极小值
    RF_discrete_point = []


    for jj in range(1, N1 + 1):
        # 速度脉冲最大取值为delta_max，最小取值-delta_max,脉冲由小逐渐变大
        Delta_V = -delta_max + 2 * delta_max * jj / N1
        for i in range(N2 + 1):
            gama = 2 * np.pi * i / N2  # gama from (0, 2pi), 从小逐渐变大
            for j in range(N3 + 1):
                alpha = -np.pi / 2 + np.pi * j / N3         # alpha from (-pi / 2, pi / 2), 从小逐渐变大
                # alpha = np.pi /2 * j / N3
                # Rf的方向单位矢量
                P = np.array([np.sin(gama) * np.cos(alpha), np.cos(gama) * np.cos(alpha), np.sin(alpha)])

                # try:
                #     temp1 = (np.sin(gama - f)) ** 2 / ((u / (p0 * Delta_V ** 2)) * (1 + e0 * np.cos(f) ** 2) - 1)
                # except ZeroDivisionError:
                #     Delta_V = 1e-10
                #     temp1 = (np.sin(gama - f)) ** 2 / ((u / (p0 * Delta_V ** 2)) * (1 + e0 * np.cos(f) ** 2) - 1)
                # temp1 = ((p0 * Delta_V ** 2) * (np.sin(gama - f)) ** 2)/ (u * (1 + e0 * np.cos(f)) ** 2 - (p0 * Delta_V ** 2))
                temp1 = (np.sin(gama-f))**2 / (u * (1 + e0 * np.cos(f))**2 / (p0 * Delta_V ** 2) - 1)
                # 可达性判据条件
                if 0 <= (np.tan(alpha))**2 <= temp1:
                    beta = np.arctan(np.tan(alpha) / np.sin(gama - f))
                    # 式（7）, 新增速度脉冲 ∆Vm
                    Delta_Vm = np.sqrt(Delta_V**2 - u * (1 + e0 * np.cos(f))**2 * (np.sin(beta))**2 / p0)

                    # 式（19）, 确定theta
                    if -2 * np.pi <= gama - f < -np.pi or 0 <= gama - f < np.pi:
                        theta = np.arccos(np.cos(gama - f) * np.cos(alpha))
                    elif -np.pi <= gama - f < 0 or np.pi <= gama - f < 2 * np.pi:
                        theta = 2 * np.pi - np.arccos(np.cos(gama - f) * np.cos(alpha))

                    # 第一个极值点
                    alpha_guess = np.pi / 2
                    # 施加脉冲后的径向和周向速度
                    v_1x = np.sqrt(u / p0) * e0 * np.sin(f) + Delta_Vm * np.cos(alpha_guess)
                    v_1y = np.sqrt(u / p0) * (1 + e0 * np.cos(f)) * np.cos(beta) + Delta_Vm * np.sin(alpha_guess)
                    # h = r0 * np.sin(f) * v_1y             # 式（17）
                    h = r0 * v_1y

                    alpha_max = Numerical_iteration_method(Delta_Vm, theta, v_1x, v_1y, u, h, alpha_guess)
                    # 计算第一个 rf 方向矢径极值
                    v_1x_max = np.sqrt(u / p0) * e0 * np.sin(f) + Delta_Vm * np.cos(alpha_max)
                    v_1y_max = np.sqrt(u / p0) * (1 + e0 * np.cos(f)) * np.cos(beta) + Delta_Vm * np.sin(alpha_max)
                    h_max = r0 * v_1y_max
                    # 式（16）
                    rf_max = h_max**2 / (u * (1 - np.cos(theta)) + h_max * v_1y_max * np.cos(theta) - h_max * v_1x_max * np.sin(theta))

                    # 第二个极值点
                    alpha_guess = -np.pi / 2
                    # 施加脉冲后的径向和周向速度
                    v_1x = np.sqrt(u / p0) * e0 * np.sin(f) + Delta_Vm * np.cos(alpha_guess)
                    v_1y = np.sqrt(u / p0) * (1 + e0 * np.cos(f)) * np.cos(beta) + Delta_Vm * np.sin(alpha_guess)
                    # h = r0 * np.sin(f) * v_1y             # 式（17）
                    h = r0 * v_1y

                    alpha_min = Numerical_iteration_method(Delta_Vm, theta, v_1x, v_1y, u, h, alpha_guess)
                    # 计算第二个 rf 方向矢径极值
                    v_1x_min = np.sqrt(u / p0) * e0 * np.sin(f) + Delta_Vm * np.cos(alpha_min)
                    v_1y_min = np.sqrt(u / p0) * (1 + e0 * np.cos(f)) * np.cos(beta) + Delta_Vm * np.sin(alpha_min)
                    h_min = r0 * v_1y_min
                    rf_min = h_min**2 / (u * (1 - np.cos(theta)) + h_min * v_1y_min * np.cos(theta) - h_min * v_1x_min * np.sin(theta))

                    RF_max.append(max(np.abs(rf_max), np.abs(rf_min)) * P)
                    RF_min.append(min(np.abs(rf_max), np.abs(rf_min)) * P)

                    # 离散失径上的点
                    # try:
                    #     RF_discrete_point_number = int(np.floor(abs(abs(rf_max) - abs(rf_min)) / delta_l) + 1)
                    #     if RF_discrete_point_number > 1:
                    #         for k in range(1, RF_discrete_point_number + 1):
                    #             RF_discrete_point.append((np.abs(rf_min) + k * delta_l) *
                    #               np.array([np.sin(gama) * np.cos(alpha), np.cos(gama) * np.cos(alpha), np.sin(alpha)]))
                    # except ValueError:
                    #     continue
                else:
                    continue

    RF_max = np.array(RF_max)
    RF_min = np.array(RF_min)
    ellipse_params = cf.Curve_fitting(RF_max, RF_min)

    # RF_combined = np.concatenate((RF_max, RF_min), axis=0)
    # RF_discrete_point = np.array(RF_discrete_point)    # 离散点

    # Excel_Writer(RF_max, RF_min)
    # plt_picture_and_save(RF_max, RF_min)
    # go_picture_and_save_html(RF_max, RF_min)
    return ellipse_params

def Numerical_iteration_method(Delta_Vm, theta, v_1x, v_1y, u, h, alpha_guess):
    def P_alpha_equation(alpha):
        eq = ((2 * u * (1 - np.cos(theta))) / (h * v_1y) - v_1x * np.sin(theta) / v_1y) * (
                    Delta_Vm * np.cos(alpha)) + np.sin(theta) * (-Delta_Vm * np.sin(alpha))
        return eq

    result = fsolve(P_alpha_equation, alpha_guess)
    return result[0]

def plt_picture_and_save(RF_max, RF_min, RF_discrete_point=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(RF_max[:, 0], RF_max[:, 1], RF_max[:, 2], c='black', marker='o', label='RF_max Points', s=1)
    # ax.plot(RF_max[:, 0], RF_max[:, 1], RF_max[:, 2], c='black', label='RF_max Line', lw=1)
    ax.scatter(RF_min[:, 0], RF_min[:, 1], RF_min[:, 2], c='b', marker='o', label='RF_min Points', s=1)
    if RF_discrete_point is not None:
        ax.scatter(RF_discrete_point[:, 0], RF_discrete_point[:, 1], RF_discrete_point[:, 2], c='r', marker='o', label='RF discrete points', s=0.5)
    # ax.plot(RF_min[:, 0], RF_min[:, 1], RF_min[:, 2], c='lightblue', label='RF_min Line', lw=1)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.legend()
    plt.show()

def go_picture_and_save_html(RF_max,RF_min):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=RF_max[:, 0], y=RF_max[:, 1], z=RF_max[:, 2], mode='markers',
                               marker=dict(size=1, color='dimgray'), name='RF_max Points'))
    fig.add_trace(go.Scatter3d(x=RF_min[:, 0], y=RF_min[:, 1], z=RF_min[:, 2], mode='markers',
                               marker=dict(size=1, color='blue'), name='RF_min Points'))
    fig.update_layout(scene=dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis'))

    # 保存为 HTML 文件
    fig.write_html('predict_domain.html')

def Excel_Writer(RF_max,RF_min):
    RF_combined = np.concatenate((RF_max, RF_min), axis=0)
    RF_max_df = pd.DataFrame(RF_max, columns=['X', 'Y', 'Z'])
    RF_min_df = pd.DataFrame(RF_min, columns=['X', 'Y', 'Z'])
    RF_combined_df = pd.DataFrame(RF_combined, columns=['X', 'Y', 'Z'])

    with pd.ExcelWriter('RF_free_data.xlsx') as writer:
        RF_max_df.to_excel(writer, sheet_name='RF_max', index=False)
        RF_min_df.to_excel(writer, sheet_name='RF_min', index=False)
        RF_combined_df.to_excel(writer, sheet_name='RF_combined_df', index=False)

# if __name__ == "__main__":
#     Reachable_Domain()
