"""
This module provides hanger zone calculations.
Reference to《基于轨道可达域的机动航天器接近威胁规避方法》——张赛

Author: Qiao beibei
Date: April 18, 2024
"""

import numpy as np
from scipy.optimize import fsolve
import math
from typing import List, Tuple, Dict, Optional
from scipy.integrate import solve_ivp
import sympy as sp

# 危险区个数及拦截窗口求解
class Time_window_of_danger_zone():
    def __init__(self,
                 R0_c=None,             # 来袭航天器位置矢量
                 V0_c=None,             # 来袭航天器速度矢量
                 R0_t=None,             # 在轨航天器位置矢量
                 V0_t=None,             # 在轨航天器速度矢量
                 c_args=None,           # 来袭航天器轨道根数
                 t_args=None,           # 在轨航天器轨道根数
                 Delta_V_c=None,        # 来袭航天器最大脉冲
                 Delta_V_t=None,        # 在轨航天器最大脉冲
                 time_step=None,        # 时间步长
                 u=3.986e14):
        self.time_step = time_step      # 时间步长
        self.u = u                      # 引力常数
        self.Delta_V_c = Delta_V_c
        self.Delta_V_t = Delta_V_t
        self.fai = 0                    # 来袭航天器在交点处的方位角fai始终为零
        self.num_td = 0                 # 危险区数量
        self.threat_t = []              # 拦截窗口
        self.azimuth = []               # 轨道周期离散时刻点的方位角
        self.o_elements = []            # 轨道周期离散时刻点的轨道六根数

        # 确保航天器的位置速度和六根数至少要提供一项, 否则警告
        assert ((R0_c is not None and V0_c is not None) or c_args is not None) and (
                    (R0_t is not None and V0_t is not None) or t_args is not None), \
            "At least one of the speed, position, and elements not be None"

        # 轨道参数[a, e, i, omega, Omega, f]
        # 已知来袭航天器位置速度，求初始轨道六根数
        if (R0_c is not None) and (V0_c is not None):
            # 确保位置速度是numpy数组，否则警告
            assert isinstance(R0_c, np.ndarray) and isinstance(V0_c, np.ndarray)
            self.R0_c = R0_c
            self.V0_c = V0_c
            # 不同类型的轨道，返回的轨道根数个数不同
            orbital_elements = self.calculate_orbital_elements(self.u, self.R0_c, self.V0_c)
            # 椭圆和双曲线轨道
            if len(orbital_elements) == 6:
                self.kinds_orbits_c = "椭圆或双曲线轨道"
                self.a_c, self.e_c, self.i_c, self.omega_c, self.Omega_c, self.f0_c = orbital_elements
                self.r_c = self.a_c * (1 - self.e_c ** 2) / (1 + self.e_c * np.cos(self.f0_c))  # 位置矢量
                self.p_c = self.a_c * (1 - self.e_c ** 2)  # 轨道半通径
            # 抛物线轨道
            elif len(orbital_elements) == 5:
                self.kinds_orbits_c = "抛物线轨道"
                self.p_c, self.i_c, self.omega_c, self.Omega_c, self.f0_c = orbital_elements
            # 圆轨道
            elif len(orbital_elements) == 4:
                self.kinds_orbits_c = "圆轨道"
                self.a_c, self.i_c, self.u_c, self.Omega_c = orbital_elements
        elif c_args is not None:
            # 已知初始轨道六根数，求来袭航天器位置速度
            self.kinds_orbits_c = "椭圆或双曲线轨道"
            self.a_c, self.e_c, self.i_c, self.omega_c, self.Omega_c, self.f0_c = c_args
            self.r_c = self.a_c * (1 - self.e_c ** 2) / (1 + self.e_c * np.cos(self.f0_c))  # 位置矢量
            self.p_c = self.a_c * (1 - self.e_c ** 2)  # 轨道半通径
            # 六根数转航天器的位置速度至惯性系
            self.R0_c, self.V0_c = self.calculate_state_information(c_args, miu=3.986e14)

        # 在轨航天器
        if (R0_t is not None) and (V0_t is not None):
            assert isinstance(R0_t, np.ndarray) and isinstance(V0_t, np.ndarray)
            self.R0_t = R0_t
            self.V0_t = V0_t
            orbital_elements = self.calculate_orbital_elements(self.u, self.R0_t, self.V0_t)
            if len(orbital_elements) == 6:
                self.kinds_orbits_t = "椭圆或双曲线轨道"
                self.a_t, self.e_t, self.i_t, self.omega_t, self.Omega_t, self.f0_t = orbital_elements
                self.r_t = self.a_t * (1 - self.e_t ** 2) / (1 + self.e_t * np.cos(self.f0_t))  # 位置矢量
                self.p_t = self.a_t * (1 - self.e_t ** 2)  # 轨道半通径
            elif len(orbital_elements) == 5:
                self.kinds_orbits_t = "抛物线轨道"
                self.p_t, self.i_t, self.omega_t, self.Omega_t, self.f0_t = orbital_elements
            elif len(orbital_elements) == 4:
                self.kinds_orbits_t = "圆轨道"
                self.a_t, self.i_t, self.u_t, self.Omega_t = orbital_elements
        elif t_args is not None:
            self.kinds_orbits_t = "椭圆或双曲线轨道"
            self.a_t, self.e_t, self.i_t, self.omega_t, self.Omega_t, self.f0_t = t_args
            self.r_t = self.a_t * (1 - self.e_t ** 2) / (1 + self.e_t * np.cos(self.f0_t))  # 位置矢量
            self.p_t = self.a_t * (1 - self.e_t ** 2)  # 轨道半通径
            # 六根数转航天器的位置速度至惯性系
            self.R0_t, self.V0_t = self.calculate_state_information(t_args, miu=3.986e14)

    def calculate_threat_all_time_domain(self):
        # 求危险区范围
        # 目标轨道的轨道周期(在轨)
        self.T_t = 2 * np.pi * np.sqrt(self.a_t ** 3 / self.u)
        # 基于时间步长T_t求出每一离散时间点处在轨航天器是否位于危险区,并记录
        for i in range(math.floor((1/self.time_step) * self.T_t)):
            i = i * self.time_step
            self.calculate_threat_time_domain(i)
        # 保留一系列数据中对应窗口的最大值和最小值
        self.threat_t, self.azimuth = self.extract_min_max()
        
        return self.threat_t, self.azimuth

    def print_information(self):
        print("=" * 64)
        print("|{:^62s}|".format(""))
        print("|{:^55s}|".format("来袭航天器轨道参数信息"))
        print("=" * 64)
        print("|{:<22s}{:>20s}".format("航天器初始位置矢量:",
                                        ", ".join(["{:.2f}".format(x) for x in self.R0_c.tolist()])))
        print("|{:<22s}{:>20s}".format("航天器初始速度矢量:",
                                        ", ".join(["{:.2f}".format(x) for x in self.V0_c.tolist()])))
        print("|{:<25s}{:>15s}".format("轨道类型:", str(self.kinds_orbits_c)))
        print("|{:<25s}{:>20.6f}".format("半长轴 - a:", self.a_c))
        print("|{:<25s}{:>20.6f}".format("离心率 - e:", self.e_c))
        print("|{:<25s}{:>20.6f}".format("轨道倾角 - i:", self.i_c))
        print("|{:<25s}{:>20.6f}".format("近心点幅角 - omega:", self.omega_c))
        print("|{:<25s}{:>20.6f}".format("升交点赤经 - Omega:", self.Omega_c))
        print("|{:<25s}{:>20.6f}".format("真近点角 - f0:", self.f0_c))
        print("|{:<24s}{:>20.2f}".format("最大速度脉冲:", self.Delta_V_c))
        print("|{:<25s}{:>20}".format("引力常数:", self.u))

        print("=" * 64)
        print("|{:^62s}|".format(""))
        print("|{:^55s}|".format("在轨航天器轨道参数信息"))
        print("=" * 64)
        print("|{:<22s}{:>20s}".format("航天器初始位置矢量:",
                                        ", ".join(["{:.2f}".format(x) for x in self.R0_t.tolist()])))
        print("|{:<22s}{:>20s}".format("航天器初始速度矢量:",
                                        ", ".join(["{:.2f}".format(x) for x in self.V0_t.tolist()])))
        print("|{:<25s}{:>15s}".format("轨道类型:", str(self.kinds_orbits_t)))
        print("|{:<25s}{:>20.6f}".format("半长轴 - a:", self.a_t))
        print("|{:<25s}{:>20.6f}".format("离心率 - e:", self.e_t))
        print("|{:<25s}{:>20.6f}".format("轨道倾角 - i:", self.i_t))
        print("|{:<25s}{:>20.6f}".format("近心点幅角 - omega:", self.omega_t))
        print("|{:<25s}{:>20.6f}".format("升交点赤经 - Omega:", self.Omega_t))
        print("|{:<25s}{:>20.6f}".format("真近点角 - f0:", self.f0_t))
        if self.Delta_V_t is not None:
            print("|{:<24s}{:>20.2f}".format("最大速度脉冲:", self.Delta_V_t))
        print("|{:<25s}{:>20}".format("引力常数:", self.u))

        print("=" * 64)
        print("|{:^62s}|".format(""))
        print("|{:^56s}|".format("危险域及拦截窗口信息"))
        print("=" * 64)
        print("|{:<25s}{:>20.6f}".format("危险区个数:", int(self.num_td)))
        print("|{:<25s}{:>20.6f}".format("目标轨道周期:", self.T_t))
        print("|{:<25s}{:>20.6f}".format("时间步长:", 0.2))
        print("|在轨航天器危险区时间窗口       ", self.threat_t)

    @staticmethod
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
        i = np.arccos(np.dot(Z, H)/h)

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

    @staticmethod
    def calculate_state_information(data, miu=3.986e14):
        """
        根据六根数推算出航天器相应的状态信息
        Args:
            data: 椭圆和双曲线转移轨道的六根数: a;e;i;omega;Omega;f
            %a: 半长轴；e：离心率；i：轨道倾角；
            %omega: 近地点幅角；Omega: 升交点经度；f: 真近点角。
            抛物线转移轨道的根数: p;i;omega;Omega;f
            %p：半矩形矩；i：轨道倾角；omega：近地点幅角；
            %Omega: 升交点经度；f: 真近点角。
            圆转移轨道的根数: a;i;u;Omega
            %a：半径；i：轨道倾角；u：纬度参数；
            %Omega: 升交点经度。
            miu: 3.986E5  # km3/s2
                 3.986E14  # m3/s2

        Returns:
            R0: 位置
            V0: 速度
        """
        if len(data) == 6:  # 椭圆和双曲线
            a = data[0]
            e = data[1]
            i = data[2]
            omega = data[3]
            Omega = data[4]
            f = data[5]
            p = abs(a * (1 - e ** 2))
            u = omega + f
        elif len(data) == 5:  # 抛物线
            p = data[0]
            i = data[1]
            omega = data[2]
            Omega = data[3]
            f = data[4]
            u = omega + f
        else:  # 圆形
            p = data[0]
            e = 0
            i = data[1]
            omega = 0
            u = data[2]
            Omega = data[3]
            f = 0

        Coordinate = p / (1 + e * np.cos(f)) * np.array([
            np.cos(Omega) * np.cos(u) - np.sin(Omega) * np.sin(u) * np.cos(i),
            np.sin(Omega) * np.cos(u) + np.cos(Omega) * np.sin(u) * np.cos(i),
            np.sin(i) * np.sin(u)
        ])

        V = (miu / p) ** 0.5 * np.array([
            -np.cos(Omega) * (np.sin(u) + e * np.sin(omega)) - np.sin(Omega) * (np.cos(u) + e * np.cos(omega)) * np.cos(i),
            -np.sin(Omega) * (np.sin(u) + e * np.sin(omega)) + np.cos(Omega) * (np.cos(u) + e * np.cos(omega)) * np.cos(i),
            np.sin(i) * (np.cos(u) + e * np.cos(omega))
        ])

        return Coordinate, V

    def calculate_latitudinal_angle(self):
        """
        计算天球面纬度幅角

        Returns:
            [u_c1, u_c2]: 来袭航天器关于引力中心对称的纬度幅角
            [u_t1, u_t2]: 在轨航天器关于引力中心对称的纬度幅角
        """
        # ->-> List[float]
        temp1 = (np.sin(self.i_t) * np.sin(self.Omega_c-self.Omega_t)) / (np.cos(self.i_t) * np.sin(self.i_c) -
                                                np.sin(self.i_t) * np.cos(self.i_c) * np.cos(self.Omega_c-self.Omega_t))
        temp2 = (np.sin(self.i_c) * np.sin(self.Omega_t-self.Omega_c)) / (np.cos(self.i_c) * np.sin(self.i_t) -
                                                np.sin(self.i_c) * np.cos(self.i_t) * np.cos(self.Omega_t-self.Omega_c))
        # 防止temp出现NaN的情况
        if np.isnan(temp1) or np.isnan(temp2):
            temp1 = temp2 = 1
        # 纬度幅角关于引力中心对称，满足 | u_x1 - u_x2 | = pi
        u_c1 = np.arctan(temp1)
        u_c2 = np.pi + u_c1
        u_t1 = np.arctan(temp2)
        u_t2 = u_t1 + np.pi

        return [u_c1, u_c2], [u_t1, u_t2]

    def calculate_number_of_hanger_area(self):
        """
        计算来袭航天器在天球面交点处的可达矢径，并判断在轨航天器的地心距是否处于危险区内.
        方位角gama为来袭航天器在交点处对应的真近点角，fai始终为0.

        Returns:
            num_td: 危险区的数量
        """
        # 求纬度幅角
        self.u_c, self.u_t = self.calculate_latitudinal_angle()
        # 求来袭航天器和在轨航天器在交点处的真近点角, f_c和f_t是一一匹配的，应处于同一象限内
        self.f_c1 = self.u_c[0] - self.omega_c
        self.f_c2 = self.u_c[1] - self.omega_c
        self.f_t1 = self.u_t[0] - self.omega_t
        self.f_t2 = self.u_t[1] - self.omega_t

        num_td = 0
        # 来袭航天器在交点1处可达矢径
        rf_max_c1, rf_min_c1 = self.rf_extreme_point('orbit_c1')
        # 来袭航天器在交点2处可达矢径
        rf_max_c2, rf_min_c2 = self.rf_extreme_point('orbit_c2')
        # 在轨航天器在交点1处轨道地心距
        r_ft1 = (self.a_t * (1 - self.e_t ** 2)) / (1 + self.e_t * np.cos(self.f_t2))
        # 在轨航天器在交点2处轨道地心距
        r_ft2 = (self.a_t * (1 - self.e_t ** 2)) / (1 + self.e_t * np.cos(self.f_t1))
        # 危险区存在数量
        if rf_min_c1 <= r_ft1 <= rf_max_c1 and rf_min_c2 <= r_ft2 <= rf_max_c2:
            num_td = 2
        elif rf_min_c1 <= r_ft1 <= rf_max_c1 or rf_min_c2 <= r_ft2 <= rf_max_c2:
            num_td = 1
        elif not (rf_min_c1 <= r_ft1 <= rf_max_c1) and not (rf_min_c2 <= r_ft2 <= rf_max_c2):
            num_td = 0
        return num_td

    def Lagrange_factor(self, sigma, delta_E):
        # 拉格朗日系数,外推轨道位置速度
        r = self.a_t + (self.r_t - self.a_t) * np.cos(delta_E) + sigma * np.sqrt(self.a_t) * np.sin(delta_E)
        F = 1 - self.a_t * (1 - np.cos(delta_E)) / self.r_t
        G = self.a_t * sigma * (1 - np.cos(delta_E)) / np.sqrt(self.u) + \
            self.r_t * np.sqrt(self.a_t / self.u) * np.sin(delta_E)
        F_t = -np.sqrt(self.u * self.a_t) * np.sin(delta_E) / (r * self.r_t)
        G_t = 1 - self.a_t * (1 - np.cos(delta_E)) / r

        return F, G, F_t, G_t

    def coordinate_J_to_g(self):
        # 从惯性系至近心点系的转移矩阵
        # 法1
        R1 = np.array([[np.cos(-self.omega_c), np.sin(-self.omega_c), 0],
                       [-np.sin(-self.omega_c), np.cos(-self.omega_c), 0],
                       [0, 0, 1]])

        R2 = np.array([[1, 0, 0],
                       [0, np.cos(-self.i_c), np.sin(-self.i_c)],
                       [0, -np.sin(-self.i_c), np.cos(-self.i_c)]])

        R3 = np.array([[np.cos(-self.Omega_c), np.sin(-self.Omega_c), 0],
                       [-np.sin(-self.Omega_c), np.cos(-self.Omega_c), 0],
                       [0, 0, 1]])
        C_Jg = np.dot(np.dot(R1, R2), R3)

        # # 法2
        # cos_Omega = np.cos(self.Omega_c)
        # sin_Omega = np.sin(self.Omega_c)
        # cos_i = np.cos(self.i_c)
        # sin_i = np.sin(self.i_c)
        # cos_omega = np.cos(self.omega_c)
        # sin_omega = np.sin(self.omega_c)
        #
        # R = np.array([
        #     [-sin_Omega * cos_i * sin_omega + cos_Omega * cos_omega,
        #      -sin_Omega * cos_i * cos_omega - cos_Omega * sin_omega,
        #      sin_Omega * sin_i],
        #     [cos_Omega * cos_i * sin_omega + sin_Omega * cos_omega,
        #      cos_Omega * cos_i * cos_omega - sin_Omega * sin_omega,
        #      -cos_Omega * sin_i],
        #     [sin_i * sin_omega,
        #      sin_i * cos_omega,
        #      cos_i]
        # ])

        return C_Jg

    def calculate_threat_time_domain(self, t):
        """
        将目标轨道周期离散，求解每一时刻处来袭航天器与在轨航天器位置是否重合

        Returns:
            self.threat_t: 危险区内的离散时间点
        """
        def delta_E_equation(delta_E):
            # 根据当前时刻的平近点角差确定偏近点角差()
            eq = delta_E + sigma * (1 - np.cos(delta_E)) / np.sqrt(self.a_t) - \
                 (1 - self.r_t / self.a_t) * np.sin(delta_E) - np.sqrt(self.u / self.a_t**3) * t
            return eq

        # 偏近点角差
        sigma = np.dot(self.R0_t, self.V0_t) / np.sqrt(self.u)

        # 当前时刻处的偏近点角差
        delta_E = fsolve(delta_E_equation, 0)
        # 航天器位置速度外推(惯性系下)
        F, G, F_t, G_t = self.Lagrange_factor(sigma, delta_E)
        R_i = F * self.R0_t + G * self.V0_t
        V_i = F_t * self.R0_t + G_t * self.V0_t
        # 惯性系至近心点坐标系的转移矩阵
        C_Jg = self.coordinate_J_to_g()
        # 近心点坐标系下航天器位置
        R_i_g = np.dot(np.linalg.inv(C_Jg), R_i)
        # 推算来袭航天器当前时刻在近心点坐标系中的空间指向
        self.gama_i = np.arctan2(R_i_g[1], R_i_g[0])
        self.fai_i = np.arctan(R_i_g[2] / np.sqrt(R_i_g[0]**2 + R_i_g[1]**2))
        # 由该空间指向推算来袭航天器当前时刻的可达矢径
        rf_max_ci, rf_min_ci = self.rf_extreme_point('orbit_ci')
        # 在轨航天器当前时刻地心距
        r_i_g = np.linalg.norm(R_i_g)
        # 判断该时刻是否为危险区
        if rf_min_ci <= r_i_g <= rf_max_ci:
            self.threat_t.append(t)
            self.azimuth.append([self.gama_i, self.fai_i, t])

    def rf_extreme_point(self, type):
        # 可达矢径计算
        Delta_Vm = beta = theta = 0
        if type == 'orbit_c1':
            temp1 = (np.sin(self.f_c1 - self.f0_c)) ** 2 / (self.u * (1 + self.e_c * np.cos(self.f0_c)) ** 2 / (self.p_c * self.Delta_V_c ** 2) - 1)
            # 可达性判据条件
            if 0 <= temp1:
                beta = np.arctan(np.tan(self.fai) / np.sin(self.f_c1 - self.f0_c))
                Delta_Vm = np.sqrt(self.Delta_V_c**2 - self.u * (1 + self.e_c * np.cos(self.f0_c)) ** 2 *
                                   (np.sin(beta)) ** 2 / self.p_c)

                if -2 * np.pi <= self.f_c1 - self.f0_c < -np.pi or 0 <= self.f_c1 - self.f0_c < np.pi:
                    theta = np.arccos(np.cos(self.f_c1 - self.f0_c) * np.cos(self.fai))
                elif -np.pi <= self.f_c1 - self.f0_c < 0 or np.pi <= self.f_c1 - self.f0_c < 2 * np.pi:
                    theta = 2 * np.pi - np.arccos(np.cos(self.f_c1 - self.f0_c) * np.cos(self.fai))
            else:
                return 0, 0

        elif type == 'orbit_c2':
            temp1 = (np.sin(self.f_c2 - self.f0_c)) ** 2 / (self.u * (1 + self.e_c * np.cos(self.f0_c)) ** 2 / (self.p_c * self.Delta_V_c ** 2) - 1)
            # 可达性判据条件
            if 0 <= temp1:
                theta = 0
                beta = np.arctan(np.tan(self.fai) / np.sin(self.f_c2 - self.f0_c))
                Delta_Vm = np.sqrt(self.Delta_V_c ** 2 - self.u * (1 + self.e_c * np.cos(self.f0_c)) ** 2 *
                                   (np.sin(beta)) ** 2 / self.p_c)

                if -2 * np.pi <= self.f_c2 - self.f0_c < -np.pi or 0 <= self.f_c2 - self.f0_c < np.pi:
                    theta = np.arccos(np.cos(self.f_c2 - self.f0_c) * np.cos(self.fai))
                elif -np.pi <= self.f_c2 - self.f0_c < 0 or np.pi <= self.f_c2 - self.f0_c < 2 * np.pi:
                    theta = 2 * np.pi - np.arccos(np.cos(self.f_c2 - self.f0_c) * np.cos(self.fai))
            else:
                return 0, 0

        elif type == 'orbit_ci':
            temp1 = (np.sin(self.gama_i - self.f0_c)) ** 2 / (self.u * (1 + self.e_c * np.cos(self.f0_c)) ** 2 / (self.p_c * self.Delta_V_c ** 2) - 1)
            a = (np.tan(self.fai_i)) ** 2
            # 可达性判据条件
            if 0 <= (np.tan(self.fai_i)) ** 2 <= temp1:
                theta = 0
                beta = np.arctan(np.tan(self.fai_i) / np.sin(self.gama_i - self.f0_c))
                # a = self.Delta_V_c ** 2
                # b = self.u * (1 + self.e_c * np.cos(self.f0_c)) ** 2 * (np.sin(0.06)) ** 2 / self.p_c
                Delta_Vm = np.sqrt(self.Delta_V_c ** 2 - self.u * (1 + self.e_c * np.cos(self.f0_c)) ** 2 *
                                   (np.sin(beta)) ** 2 / self.p_c)

                if -2 * np.pi <= self.gama_i - self.f0_c < -np.pi or 0 <= self.gama_i - self.f0_c < np.pi:
                    theta = np.arccos(np.cos(self.gama_i - self.f0_c) * np.cos(self.fai_i))
                elif -np.pi <= self.gama_i - self.f0_c < 0 or np.pi <= self.gama_i - self.f0_c < 2 * np.pi:
                    theta = 2 * np.pi - np.arccos(np.cos(self.gama_i - self.f0_c) * np.cos(self.fai_i))
            else:
                return 0, 0

        # 第一个极值点
        alpha_guess = np.pi / 2
        # 施加脉冲后的径向和周向速度
        v_1x = np.sqrt(self.u / self.p_c) * self.e_c * np.sin(self.f0_c) + Delta_Vm * np.cos(alpha_guess)
        v_1y = np.sqrt(self.u / self.p_c) * (1 + self.e_c * np.cos(self.f0_c)) * np.cos(
            beta) + Delta_Vm * np.sin(alpha_guess)
        h = self.r_c * v_1y

        alpha_max = self.Numerical_iteration_method(Delta_Vm, theta, v_1x, v_1y, h, alpha_guess)
        # 计算第一个 rf 方向矢径极值
        v_1x_max = np.sqrt(self.u / self.p_c) * self.e_c * np.sin(self.f0_c) + Delta_Vm * np.cos(alpha_max)
        v_1y_max = np.sqrt(self.u / self.p_c) * (1 + self.e_c * np.cos(self.f0_c)) * np.cos(
            beta) + Delta_Vm * np.sin(alpha_max)
        h_max = self.r_c * v_1y_max
        # 式（16）
        rf_max = h_max ** 2 / (
                self.u * (1 - np.cos(theta)) + h_max * v_1y_max * np.cos(theta) - h_max * v_1x_max * np.sin(theta))

        # 第二个极值点
        alpha_guess = -np.pi / 2
        v_1x = np.sqrt(self.u / self.p_c) * self.e_c * np.sin(self.f0_c) + Delta_Vm * np.cos(alpha_guess)
        v_1y = np.sqrt(self.u / self.p_c) * (1 + self.e_c * np.cos(self.f0_c)) * np.cos(
            beta) + Delta_Vm * np.sin(alpha_guess)
        h = self.r_c * v_1y

        alpha_min = self.Numerical_iteration_method(Delta_Vm, theta, v_1x, v_1y, h, alpha_guess)
        v_1x_min = np.sqrt(self.u / self.p_c) * self.e_c * np.sin(self.f0_c) + Delta_Vm * np.cos(alpha_min)
        v_1y_min = np.sqrt(self.u / self.p_c) * (1 + self.e_c * np.cos(self.f0_c)) * np.cos(
            beta) + Delta_Vm * np.sin(alpha_min)
        h_min = self.r_c * v_1y_min
        rf_min = h_min ** 2 / (
                self.u * (1 - np.cos(theta)) + h_min * v_1y_min * np.cos(theta) - h_min * v_1x_min * np.sin(
            theta))

        rf_max = np.abs(rf_max)
        rf_min = np.abs(rf_min)
        if rf_max < rf_min:
            num = rf_min
            rf_min = rf_max
            rf_max = num

        return rf_max, rf_min

    def Numerical_iteration_method(self, Delta_Vm, theta, v_1x, v_1y, h, alpha_guess):
        def P_fai_equation(alpha):
            eq = ((2 * self.u * (1 - np.cos(theta))) / (h * v_1y) - v_1x * np.sin(theta) / v_1y) * (
                        Delta_Vm * np.cos(alpha)) + np.sin(theta) * (-Delta_Vm * np.sin(alpha))
            return eq

        result = fsolve(P_fai_equation, alpha_guess)
        return result[0]

    def extract_min_max(self):
        """
        求出处于危险区内一系列离散时间点的最大值和最小值
        arg:
            threat_t: 在轨航天器危险区内时间窗口

        Returns:
            min_max_pairs: 存储与危险区数目对应的在轨航天器的时间窗口
            min_max_azimuth: 存储与时间点对应的来袭航天器方位角
            min_max_o_elements: 存储与时间点对应的在轨航天器轨道根数
        """
        if not self.threat_t:
            return []

        min_max_pairs = []
        min_max_azimuth = []
        start = 0
        while start < len(self.threat_t):
            if start + 1 < len(self.threat_t) and self.threat_t[start + 1] - self.threat_t[start] > 10:
                min_max_pairs.append((self.threat_t[0], self.threat_t[start]))
                min_max_pairs.append((self.threat_t[start + 1], self.threat_t[len(self.threat_t) - 1]))
                min_max_azimuth.append((self.azimuth[0], self.azimuth[start]))
                min_max_azimuth.append((self.azimuth[start + 1], self.azimuth[len(self.threat_t) - 1]))
                break
            elif start == len(self.threat_t) - 1:
                min_max_pairs.append((self.threat_t[0], self.threat_t[len(self.threat_t) - 1]))
                min_max_azimuth.append((self.azimuth[0], self.azimuth[len(self.threat_t) - 1]))
            start += 1

        return min_max_pairs, min_max_azimuth

    def calculation_spacecraft_status(self, t):
        """
        :param t: 航天器轨道状态信息外推时间
        :return: 航天器轨道状态信息
        """
        def delta_E_equation_t(delta_E):
            # 根据当前时刻的平近点角差确定偏近点角差()
            eq = delta_E + sigma_t * (1 - np.cos(delta_E)) / np.sqrt(self.a_t) - \
                 (1 - self.r_t / self.a_t) * np.sin(delta_E) - np.sqrt(self.u / self.a_t**3) * t
            return eq

        def delta_E_equation_c(delta_E):
            # 根据当前时刻的平近点角差确定偏近点角差()
            eq = delta_E + sigma_c * (1 - np.cos(delta_E)) / np.sqrt(self.a_c) - \
                 (1 - self.r_c / self.a_c) * np.sin(delta_E) - np.sqrt(self.u / self.a_c**3) * t
            return eq

        def Lagrange_factor_c(sigma, delta_E):
            # 拉格朗日系数,外推轨道位置速度
            r = self.a_c + (self.r_c - self.a_c) * np.cos(delta_E) + sigma * np.sqrt(self.a_c) * np.sin(delta_E)
            F = 1 - self.a_c * (1 - np.cos(delta_E)) / self.r_c
            G = self.a_c * sigma * (1 - np.cos(delta_E)) / np.sqrt(self.u) + \
                self.r_c * np.sqrt(self.a_c / self.u) * np.sin(delta_E)
            F_c = -np.sqrt(self.u * self.a_c) * np.sin(delta_E) / (r * self.r_c)
            G_c = 1 - self.a_c * (1 - np.cos(delta_E)) / r

            return F, G, F_c, G_c
        # 偏近点角差

        sigma_t = np.dot(self.R0_t, self.V0_t) / np.sqrt(self.u)
        sigma_c = np.dot(self.R0_c, self.V0_c) / np.sqrt(self.u)
        # 当前时刻处的偏近点角差
        delta_E_t = fsolve(delta_E_equation_t, 0)
        delta_E_c = fsolve(delta_E_equation_c, 0)
        # print(self.R0_t, self.V0_t, self.R0_c, self.V0_c)
        # print(delta_E_t, sigma_c, self.a_c, self.r_c / self.a_c, np.sqrt(self.u / self.a_c ** 3) * t)
        #
        # 航天器位置速度外推(惯性系下)
        F, G, F_t, G_t = self.Lagrange_factor(sigma_t, delta_E_t)
        R_i_t = F * self.R0_t + G * self.V0_t
        V_i_t = F_t * self.R0_t + G_t * self.V0_t

        F, G, F_c, G_c = Lagrange_factor_c(sigma_c, delta_E_c)
        R_i_c = F * self.R0_c + G * self.V0_c
        V_i_c = F_c * self.R0_c + G_t * self.V0_c


        # 惯性系至近心点坐标系的转移矩阵
        C_Jg = self.coordinate_J_to_g()
        # 近心点坐标系下航天器位置
        R_i_t = np.dot(np.linalg.inv(C_Jg), R_i_t)
        V_i_t = np.dot(np.linalg.inv(C_Jg), V_i_t)
        R_i_c = np.dot(np.linalg.inv(C_Jg), R_i_c)
        V_i_c = np.dot(np.linalg.inv(C_Jg), V_i_c)

        return np.array([R_i_c, V_i_c]).ravel(), np.array([R_i_t, V_i_t]).ravel()

class Danger_index_and_TW_matching_index():
    def __init__(self, TW_danger_zone):
        self.danger_zone = TW_danger_zone.threat_t   # 在轨航天器时间窗口
        self.num_td =TW_danger_zone.num_td           # 危险区数量
        self.T_t = TW_danger_zone.T_t                # 在轨航天器轨道周期
        self.u = TW_danger_zone.u
        # 截面方位角
        self.azimuth = TW_danger_zone.azimuth  # type: List[Tuple[List[float, float], List[float, float]]]
        self.t_w = []                                # 来袭航天器时间窗口
        # 求解基于危险区比例的评价系数, 在轨航天器处于危险区的时间长度 / 轨道周期
        if self.num_td == 2:
            self.ksi_D = (self.danger_zone[0][1] - self.danger_zone[0][0] + self.danger_zone[1][1] -
                          self.danger_zone[1][0]) / self.T_t
        elif self.num_td == 1:
            self.ksi_D = (self.danger_zone[0][1] - self.danger_zone[0][0]) / self.T_t
        elif self.num_td == 0:
            self.ksi_D = 0
        # 来袭航天器轨道六根数
        self.a, self.e, self.i, self.omega, self.Omega, self.f0 = TW_danger_zone.a_c, TW_danger_zone.e_c, \
                               TW_danger_zone.i_c, TW_danger_zone.omega_c, TW_danger_zone.Omega_c, TW_danger_zone.f0_c
        self.p = TW_danger_zone.p_c
        self.r = TW_danger_zone.r_c
        self.Delta_V_c = TW_danger_zone.Delta_V_c
        # 来袭航太器在截面处的时间窗口
        self.calculate_danger_time(TW_danger_zone)

    def print_information(self):
        print("|来袭航天器危险区时间窗口       [({}, {}), ({}, {})]".format(
            *[round(num, 2) for sublist in self.t_w for num in sublist]))
        print("|{:<22s}{:>20.6f}".format("在轨航天器危险系数 - ksi_D:", self.ksi_D))

    def calculate_danger_time(self, TW_danger_zone):
        # 真近角与偏近角转换
        def E_w_equation(E):
            eq = np.tan(E / 2) - np.sqrt((1 - data[1]) / (1 + data[1])) * np.tan(data[5] / 2)
            return eq

        def E_gama1_equation(E):
            eq = np.tan(E / 2) - np.sqrt((1 - data[1]) / (1 + data[1])) * np.tan(f_gama / 2)
            return eq

        for i in range(self.num_td):
            V_w1 = (np.sqrt((self.u/self.p) * (1 + self.e**2 + 2 * self.e * np.cos(self.f0))) - self.Delta_V_c) / \
                   np.sqrt(1 + self.e**2 + 2 * self.e * np.cos(self.f0))
            V_w1 = V_w1 * np.array([self.e * np.sin(self.f0), 1 + self.e * np.cos(self.f0), 0])
            R_w1 = np.array([self.r * np.cos(self.f0), self.r * np.sin(self.f0), 0])
            # 转移轨道六根数
            # data = [a, e, i, omega, Omega, f]
            data = TW_danger_zone.calculate_orbital_elements(self.u, R_w1, V_w1)
            # 截面game1在转移轨道处对应的真近点角
            f_gama = self.azimuth[i][0][0] - data[5]
            # 来袭机动平台偏近角
            E_w1 = fsolve(E_w_equation, data[5])[0]
            # gama1截面处对应的偏近角
            E_gama = fsolve(E_gama1_equation, f_gama)[0]

            # 计算需要跨越的轨道周期数
            periods_crossed = math.ceil((E_w1 - E_gama) / (2 * math.pi))
            # 调整E2以使其大于E1
            E_gama = E_gama + periods_crossed * 2 * math.pi

            # 来袭航天器抵达危险区的最小时间
            t_w_min = np.sqrt(data[0] ** 3 / self.u) * (
                    (E_gama - E_w1) - (data[1] * np.sin(E_gama) - data[1] * np.sin(E_w1)))
            a = 2*np.pi * np.sqrt(data[0]**3/self.u)
            # if t_w_min < 0 :
            #     t_w_min = t_w_min + 2*np.pi * np.sqrt(data[0]**3/self.u)

            V_w2 = (np.sqrt((self.u / self.p) * (1 + self.e ** 2 + 2 * self.e * np.cos(self.f0))) + self.Delta_V_c) / \
                   np.sqrt(1 + self.e ** 2 + 2 * self.e * np.cos(self.f0)) * \
                   np.array([self.e * np.sin(self.f0), 1 + self.e * np.cos(self.f0), 0])
            R_w2 = np.array([self.r * np.cos(self.f0), self.r * np.sin(self.f0), 0])
            data = TW_danger_zone.calculate_orbital_elements(self.u, R_w2, V_w2)
            f_gama = self.azimuth[i][1][0] - data[5]
            E_w2 = fsolve(E_w_equation, data[5]/2)[0]
            E_gama = fsolve(E_gama1_equation, f_gama)[0]

            # 计算需要跨越的轨道周期数
            periods_crossed = math.ceil((E_w2 - E_gama) / (2 * math.pi))
            # 调整E2以使其大于E1
            E_gama = E_gama + periods_crossed * 2 * math.pi

            t_w_max = np.sqrt(data[0] ** 3 / self.u) * (
                        (E_gama - data[1] * np.sin(E_gama)) - (E_w2 - data[1] * np.sin(E_w2)))
            b = 2*np.pi * np.sqrt(data[0]**3/self.u)
            # if t_w_max < 0:
            #     t_w_max = t_w_max + 2*np.pi * np.sqrt(data[0]**3/self.u)
            self.t_w.append([t_w_min, t_w_max])

class Clohessy_Wiltshire():
    def __init__(self,R0_c=None, V0_c=None, R0_t=None, V0_t=None):
        self.R0_c = R0_c
        self.V0_c = V0_c
        self.R0_t = R0_t
        self.V0_t = V0_t
        self.u = 3.986e14


    def State_transition_matrix(self, t):
        self.t = t
        # GEO轨道半径
        R = 42164000
        # target轨道半径
        # r = R + np.sqrt(self.R0_t[0]**2 + self.R0_t[0]**2 + self.R0_t[0]**2)
        r = R
        # target轨道平均角速度
        omega = math.sqrt(self.u / (r ** 3))    # 轨道平均角速度

        tau = omega * self.t
        s = np.sin(tau)
        c = np.cos(tau)
        elements = [
            [4 - 3*c, 0, 0, s/omega , 2*(1 - c)/omega, 0],
            [6*(s - tau), 1, 0, -2*(1 - c)/omega, 4*s/omega - 3*tau, 0],
            [0, 0, c, 0 , 0, s/omega],
            [3*omega*s, 0 ,0, c, 2*s, 0],
            [6*omega*(c - 1), 0, 0, -2*s, 4*c - 3, 0],
            [0, 0 , -omega*s, 0, 0, c]
        ]

        matrix = np.array(elements)
        state_c = np.array([self.R0_c,self.V0_c]).ravel()
        state_t = np.array([self.R0_t, self.V0_t]).ravel()
        state_c_new = np.dot(matrix, state_c)
        state_t_new = np.dot(matrix, state_t)

        return state_c_new, state_t_new

class Numerical_calculation_method():
    def __init__(self, R0_c=None, V0_c=None, R0_t=None, V0_t=None):
        self.R0_c = R0_c
        self.V0_c = V0_c
        self.R0_t = R0_t
        self.V0_t = V0_t
        self.u = 3.986e5
        self.pursuer_initial_state = np.concatenate((self.R0_c, self.V0_c))
        self.escaper_initial_state = np.concatenate((self.R0_t, self.V0_t))

    @staticmethod
    def orbit_ode(t, X, Tmax, direction):
        Radius = 6378
        mu = 398600
        J2 = 0
        m = 300
        r = 35786
        # r = np.sqrt((X[0]**2 + X[1]**2 + X[2]**2))
        omega = math.sqrt(mu / (r ** 3))  # 轨道平均角速度
        I = np.array([1, 0, 0])
        J = np.array([0, 1, 0])
        K = np.array([0, 0, 1])

        # 生成1*3向量，作为J2摄动的3个分量，X(1:3)为位置，X(4:6)为速度
        pJ2 = ((3 * J2 * mu * Radius ** 2) / (2 * r ** 4)) * (((X[0] / r) * (5 * (X[2] / r) ** 2 - 1)) * I +
                                                              ((X[1] / r) * (5 * (X[2] / r) ** 2 - 1)) * J +
                                                              ((X[2] / r) * (5 * (X[2] / r) ** 2 - 3)) * K)

        # 推力分量
        a_Tr = Tmax * math.cos(direction[0]) * math.cos(direction[1]) / m
        a_Tn = Tmax * math.cos(direction[0]) * math.sin(direction[1]) / m
        a_Tt = Tmax * math.sin(direction[0]) / m

        # C-W动力学方程，返回状态的导数
        dydt = [X[3], X[4], X[5],
                (2 * omega * X[4] + 3 * omega ** 2 * X[0] + a_Tr + pJ2[0]),
                (-2 * omega * X[3] + a_Tn + pJ2[1]),
                (-omega ** 2 * X[2] + a_Tt + pJ2[2])]
        return dydt

    def numerical_calculation(self, t):
        Tmax = 0
        direction = [1, 1]
        extra_params = (Tmax, direction)
        time_step = 50
        t_eval = np.arange(0, t + time_step, time_step)
        solution1 = solve_ivp(self.orbit_ode, (0, t), self.pursuer_initial_state, args=extra_params, method='RK45', t_eval=t_eval)
        solution2 = solve_ivp(self.orbit_ode, (0, t), self.escaper_initial_state, args=extra_params, method='RK45', t_eval=t_eval)
        self.R0_c = solution1.y[:3, -1]
        self.V0_c = solution1.y[3:, -1]
        self.R0_t = solution2.y[:3, -1]
        self.V0_t = solution2.y[3:, -1]

        state_c = np.array([self.R0_c, self.V0_c]).ravel()
        state_t = np.array([self.R0_t, self.V0_t]).ravel()

        return state_c, state_t
