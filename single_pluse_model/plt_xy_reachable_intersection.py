import RD_single_pulse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 绘制航天器在惯性系下的轨道
def plot_orbit(a, e, i, Omega, omega, nu):

    # 将角度转换为弧度
    i = np.radians(i)
    Omega = np.radians(Omega)
    omega = np.radians(omega)
    nu = np.radians(nu)

    # 定义轨道参数
    p = a * (1 - e**2)  # 半通径

    # 参数化轨道
    theta = np.linspace(0, 2 * np.pi, 1000)
    r = p / (1 + e * np.cos(theta))

    # 绘制轨道在轨道平面中的位置
    x_orbit = r * np.cos(theta)
    y_orbit = r * np.sin(theta)
    z_orbit = np.zeros_like(x_orbit)

    # 旋转到地心惯性系
    rotation_matrix = np.array([
        [np.cos(Omega)*np.cos(omega) - np.sin(Omega)*np.sin(omega)*np.cos(i), -np.cos(Omega)*np.sin(omega) - np.sin(Omega)*np.cos(omega)*np.cos(i), np.sin(Omega)*np.sin(i)],
        [np.sin(Omega)*np.cos(omega) + np.cos(Omega)*np.sin(omega)*np.cos(i), -np.sin(Omega)*np.sin(omega) + np.cos(Omega)*np.cos(omega)*np.cos(i), -np.cos(Omega)*np.sin(i)],
        [np.sin(omega)*np.sin(i), np.cos(omega)*np.sin(i), np.cos(i)]
    ])

    xyz_orbit = np.dot(rotation_matrix, np.array([x_orbit, y_orbit, z_orbit]))

    # 绘制轨道
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xyz_orbit[0], xyz_orbit[1], xyz_orbit[2], label='Orbit')
    ax.scatter([0], [0], [0], color='yellow', label='Central Body (e.g., Earth)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Orbit in 3D')
    ax.legend()
    plt.show()

# 绘制航天器在惯性系下的xoy面轨道
def plot_orbit_2d(a, e, omega, nu):
    # 将角度转换为弧度
    omega = np.radians(omega)
    nu = np.radians(nu)

    # 定义轨道参数
    p = a * (1 - e**2)  # 半通径

    # 参数化轨道
    theta = np.linspace(0, 2 * np.pi, 1000)
    r = p / (1 + e * np.cos(theta))

    # 轨道平面中的位置
    x_orbit = r * np.cos(theta)
    y_orbit = r * np.sin(theta)

    # 旋转到正确的位置
    x_rot = x_orbit * np.cos(omega) - y_orbit * np.sin(omega)
    y_rot = x_orbit * np.sin(omega) + y_orbit * np.cos(omega)

    # 绘制轨道
    plt.figure(figsize=(8, 8))
    plt.plot(x_rot, y_rot, label='Orbit')
    plt.scatter([0], [0], color='yellow', label='Central Body (e.g., Earth)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Orbit in 2D (x-y plane)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# 绘制椭圆
def Plt_ellipse(data):
    data = np.array(data)
    # 拆分数据
    ellipse1_params = data[0]
    ellipse2_params = data[1]

    # 提取第一个椭圆的参数
    xc1, yc1, a1, b1, theta1 = ellipse1_params

    # 提取第二个椭圆的参数
    xc2, yc2, a2, b2, theta2 = ellipse2_params

    # 生成椭圆的点
    t = np.linspace(0, 2 * np.pi, 100)

    # 第一个椭圆
    Ell1 = np.array([a1 * np.cos(t), b1 * np.sin(t)])
    R1 = np.array([[np.cos(theta1), -np.sin(theta1)], [np.sin(theta1), np.cos(theta1)]])
    Ell_rot1 = np.dot(R1, Ell1)

    # 第二个椭圆
    Ell2 = np.array([a2 * np.cos(t), b2 * np.sin(t)])
    R2 = np.array([[np.cos(theta2), -np.sin(theta2)], [np.sin(theta2), np.cos(theta2)]])
    Ell_rot2 = np.dot(R2, Ell2)

    # 绘制椭圆
    plt.plot(xc1 + Ell_rot1[0, :], yc1 + Ell_rot1[1, :], label=f'Fitted Ellipse 1: a={a1:.2f}, b={b1:.2f}, θ={theta1:.2f}')
    plt.plot(xc2 + Ell_rot2[0, :], yc2 + Ell_rot2[1, :], label=f'Fitted Ellipse 2: a={a2:.2f}, b={b2:.2f}, θ={theta2:.2f}')

    # 设置图形参数
    plt.legend()
    plt.title('Fitted Ellipses')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

# 将椭圆和在轨航天器xoy面轨道绘制在一起，判断时间窗口
def plot_combined(data, a0, e, omega_deg, nu):
    data = np.array(data)
    # 拆分数据
    ellipse1_params = data[0]
    ellipse2_params = data[1]

    # 提取第一个椭圆的参数
    xc1, yc1, a1, b1, theta1 = ellipse1_params

    # 提取第二个椭圆的参数
    xc2, yc2, a2, b2, theta2 = ellipse2_params

    # 生成椭圆的点
    t = np.linspace(0, 2 * np.pi, 100)

    # 第一个椭圆
    Ell1 = np.array([a1 * np.cos(t), b1 * np.sin(t)])
    R1 = np.array([[np.cos(theta1), -np.sin(theta1)], [np.sin(theta1), np.cos(theta1)]])
    Ell_rot1 = np.dot(R1, Ell1)

    # 第二个椭圆
    Ell2 = np.array([a2 * np.cos(t), b2 * np.sin(t)])
    R2 = np.array([[np.cos(theta2), -np.sin(theta2)], [np.sin(theta2), np.cos(theta2)]])
    Ell_rot2 = np.dot(R2, Ell2)

    # 设置绘图
    plt.figure(figsize=(10, 10))

    # 绘制椭圆
    plt.plot(xc1 + Ell_rot1[0, :], yc1 + Ell_rot1[1, :], color='black', label=f'Fitted Ellipse 1: a={a1:.2f}, b={b1:.2f}, θ={theta1:.2f}')
    plt.plot(xc2 + Ell_rot2[0, :], yc2 + Ell_rot2[1, :], color='black', label=f'Fitted Ellipse 2: a={a2:.2f}, b={b2:.2f}, θ={theta2:.2f}')

    # 轨道参数
    omega = np.radians(omega_deg)

    # 定义轨道参数
    p = a0 * (1 - e**2)  # 半通径

    # 参数化轨道
    theta = np.linspace(0, 2 * np.pi, 1000)
    r = p / (1 + e * np.cos(theta))

    # 轨道平面中的位置
    x_orbit = r * np.cos(theta)
    y_orbit = r * np.sin(theta)

    # 旋转到正确的位置
    x_rot = x_orbit * np.cos(omega) - y_orbit * np.sin(omega)
    y_rot = x_orbit * np.sin(omega) + y_orbit * np.cos(omega)

    # 绘制轨道
    plt.plot(x_rot, y_rot, label='Orbit', color='blue')
    plt.scatter([0], [0], color='yellow', label='Central Body (e.g., Earth)')

    # 设置图形参数
    plt.legend()
    plt.title('Fitted Ellipses and Orbit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

c_args = [10 ** 7, 0.2, np.deg2rad(20), 0, 0, np.deg2rad(30)],
t_args = [10 ** 7, 0.15, np.deg2rad(60), np.deg2rad(-90), np.deg2rad(90), 0],


a = RD_single_pulse.Incoming_parameters([10 ** 7, 0.2, np.deg2rad(20), 0, 0 ,np.deg2rad(30)],300)

# Plt_ellipse(a)
plot_combined(a,10 ** 7,0.15,np.deg2rad(-90),3.986e14)



# plot_orbit(10 ** 7, 0.15, np.deg2rad(60), np.deg2rad(90), np.deg2rad(-90), 3.986e14)
