import numpy as np
import RD_single_pulse
from tqdm import tqdm
import pandas as pd

def parse_line(line):
    try:
        # 拆分位置、速度和燃料数据
        prefix, data = line.split(': ')

        # 通过检查具体格式来解析位置和速度
        position_start = data.find('[') + 1
        position_end = data.find(']')
        position_str = data[position_start:position_end]

        velocity_start = data.find('[', position_end) + 1
        velocity_end = data.find(']', velocity_start)
        velocity_str = data[velocity_start:velocity_end]

        # 剩余部分是燃料数据
        fuel_str = data[velocity_end + 1:].strip()

        # 转换为浮点数
        position = [float(x) for x in position_str.split()]
        velocity = [float(x) for x in velocity_str.split()]
        fuel = float(fuel_str)

        return position + velocity + [fuel]
    except ValueError as e:
        print(f"Error parsing line: {line}")
        raise e

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

def save_to_csv(data, output_file):
    flattened_data = []
    data = data.reshape(data.shape[0], -1)

    # for input_index, points in enumerate(data):
    #     num_points = points.shape[0]
    #     for point_index in range(num_points):
    #         x, y, z = points[point_index]
    #         flattened_data.append([input_index, point_index, x, y, z])

    # 转换为DataFrame
    df = pd.DataFrame(data, columns=['xc', 'yc', 'a', 'b', 'theta', 'xc', 'yc', 'a', 'b', 'theta'])
    # 保存到Excel文件
    df.to_csv(output_file, index=False)

    print(f"Data saved to {output_file}")

def main():
    # 读取文件并解析每一行
    input_file = 'spacecraft_state.txt'
    data = []

    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # 跳过空行
                data.append(parse_line(line))

    # 转换为numpy数组
    data = np.array(data)
    all_input = []
    for input in data:
        output = []
        output = calculate_orbital_elements(3.986e14, input[0:3], input[3:6])
        output.append(input[6])
        all_input.append(output)
    df = pd.DataFrame(all_input, columns=['a', 'e', 'i', 'omega', 'Omega', 'f', 'fuel'])
    selected_columns = df[['a', 'e', 'i', 'f', 'fuel']]
    # 保存到csv文件
    selected_columns.to_csv('all_input.csv', index=False)
    all_input = np.array(all_input)

    all_output = []
    with tqdm(total=all_input.shape[0], desc='总进度') as pbar_total:
        for ip in all_input:
            pbar_total.update(1)
            all_output.append(RD_single_pulse.Incoming_parameters(ip, ip[6]))
        all_output = np.array(all_output)

    output_file = 'output_data.csv'
    save_to_csv(all_output, output_file)

if __name__ == "__main__":
    main()
