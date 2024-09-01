# import numpy as np
# from scipy.linalg import svd
# import pandas as pd
#
# def fit_ellipse(x, y):
#     D = np.vstack([x**2, x*y, y**2, x, y, np.ones_like(x)]).T
#     U, S, Vt = svd(D)
#     V = Vt.T
#     A = V[:, -1]
#     return A
#
# def ellipse_params(A):
#     b, c, d, f, g, a = A[1] / 2, A[2], A[3] / 2, A[4] / 2, A[5], A[0]
#     num = b*b - a*c
#     x0 = (c*d - b*f) / num
#     y0 = (a*f - b*d) / num
#     term = 2*(a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g)
#     width = np.sqrt(term / ((b*b - a*c)*(np.sqrt((a - c)**2 + 4*b*b) - (a + c))))
#     height = np.sqrt(term / ((b*b - a*c)*(-np.sqrt((a - c)**2 + 4*b*b) - (a + c))))
#     phi = 0.5 * np.arctan(2*b / (a - c))
#     return (x0, y0, width, height, np.degrees(phi))
#
# # 假设你有一组点集 (x, y)
# df = pd.read_excel("RF_free_data1.xlsx", sheet_name="RF_combined_df")
# df = df[['X', 'Y']]
# x = np.array(df['X'])
# y = np.array(df['Y'])
# A = fit_ellipse(x, y)
# ellipse_params(A)
#

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from scipy.linalg import svd
# import pandas as pd
#
# # 椭圆拟合函数
# def fit_ellipse(x, y):
#     D = np.vstack([x**2, x*y, y**2, x, y, np.ones_like(x)]).T
#     U, S, Vt = svd(D)
#     V = Vt.T
#     A = V[:, -1]
#     return A
#
# def ellipse_params(A):
#     b, c, d, f, g, a = A[1] / 2, A[2], A[3] / 2, A[4] / 2, A[5], A[0]
#     num = b*b - a*c
#     if num == 0:
#         return (np.nan, np.nan, np.nan, np.nan, np.nan)
#     x0 = (c*d - b*f) / num
#     y0 = (a*f - b*d) / num
#     term = 2*(a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g)
#     width = np.sqrt(term / ((b*b - a*c)*(np.sqrt((a - c)**2 + 4*b*b) - (a + c))))
#     height = np.sqrt(term / ((b*b - a*c)*(-np.sqrt((a - c)**2 + 4*b*b) - (a + c))))
#     phi = 0.5 * np.arctan(2*b / (a - c))
#     return (x0, y0, width, height, np.degrees(phi))
#
# # 设置字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#
# # 示例数据集，替换为实际的点集数据
# # 这里假设你已经有点集 (x, y)
# df = pd.read_excel("RF_free_data1.xlsx", sheet_name="RF_combined_df")
# df.drop_duplicates(subset=['X', 'Y'], keep='first', inplace=True)
# df = df[['X', 'Y']]
# x = np.array(df['X'])
# y = np.array(df['Y'])
#
# # 聚类分割
# points = np.column_stack((x, y))
# kmeans = KMeans(n_clusters=2, n_init=10).fit(points)
# labels = kmeans.labels_
#
# # 分离两个椭圆的点集
# x_outer = x[labels == 0]
# y_outer = y[labels == 0]
# x_inner = x[labels == 1]
# y_inner = y[labels == 1]
#
# # 拟合外部椭圆
# A_outer = fit_ellipse(x_outer, y_outer)
# params_outer = ellipse_params(A_outer)
# print("外部椭圆参数:", params_outer)
#
# # 拟合内部椭圆
# A_inner = fit_ellipse(x_inner, y_inner)
# params_inner = ellipse_params(A_inner)
# print("内部椭圆参数:", params_inner)
#
# # 可视化拟合结果
# theta = np.linspace(0, 2*np.pi, 100)
#
# # 外部椭圆
# ellipse_outer = np.array([params_outer[2] * np.cos(theta), params_outer[3] * np.sin(theta)])
# R_outer = np.array([[np.cos(np.radians(params_outer[4])), -np.sin(np.radians(params_outer[4]))],
#                     [np.sin(np.radians(params_outer[4])), np.cos(np.radians(params_outer[4]))]])
# ellipse_outer = R_outer @ ellipse_outer
# ellipse_outer[0, :] += params_outer[0]
# ellipse_outer[1, :] += params_outer[1]
#
# # 内部椭圆
# ellipse_inner = np.array([params_inner[2] * np.cos(theta), params_inner[3] * np.sin(theta)])
# R_inner = np.array([[np.cos(np.radians(params_inner[4])), -np.sin(np.radians(params_inner[4]))],
#                     [np.sin(np.radians(params_inner[4])), np.cos(np.radians(params_inner[4]))]])
# ellipse_inner = R_inner @ ellipse_inner
# ellipse_inner[0, :] += params_inner[0]
# ellipse_inner[1, :] += params_inner[1]
#
# plt.plot(x_outer, y_outer, 'b.', label='外部椭圆点')
# plt.plot(x_inner, y_inner, 'r.', label='内部椭圆点')
# plt.plot(ellipse_outer[0, :], ellipse_outer[1, :], 'b-', label='拟合外部椭圆')
# plt.plot(ellipse_inner[0, :], ellipse_inner[1, :], 'r-', label='拟合内部椭圆')
# plt.xlabel('x/km')
# plt.ylabel('y/km')
# plt.legend()
# plt.show()

# df = pd.read_excel("RF_free_data1.xlsx", sheet_name="RF_combined_df")
# df = df[['X', 'Y']]
# df.drop_duplicates(subset=['X', 'Y'], keep='first', inplace=True)
# x = np.array(df['X'])
# y = np.array(df['Y'])
# points = np.column_stack((x, y))

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
# import pandas as pd
# from scipy.optimize import least_squares
#
# # Load your points (x, y) here
# df = pd.read_excel("RF_free_data1.xlsx", sheet_name="RF_combined_df")
# df = df[['X', 'Y']]
# df.drop_duplicates(subset=['X', 'Y'], keep='first', inplace=True)
# x = np.array(df['X'])
# y = np.array(df['Y'])
# points = np.column_stack((x, y))
#
# # Apply DBSCAN to find clusters
# dbscan = DBSCAN(eps=1e6, min_samples=10)
# labels = dbscan.fit_predict(points)
#
# # Check if any clusters were found
# if len(set(labels)) <= 1:  # Only noise or a single cluster found
#     raise ValueError("DBSCAN did not find any clusters. Adjust 'eps' and 'min_samples' parameters.")
#
# # Calculate the centers of the detected clusters
# unique_labels = set(labels)
# centers = []
# for label in unique_labels:
#     if label != -1:  # Exclude noise
#         cluster_points = points[labels == label]
#         center = cluster_points.mean(axis=0)
#         centers.append(center)
#
# # Calculate distances from each point to the nearest center
# filtered_points_list = []
# for center in centers:
#     distances = np.linalg.norm(points - center, axis=1)
#     threshold = np.median(distances)
#     filtered_points = points[distances < threshold]
#     filtered_points_list.append(filtered_points)
#
# # Ensure that filtered_points_list is a list of 2D arrays
# if not filtered_points_list:
#     raise ValueError("No points found within the specified thresholds.")
#
# filtered_points = np.vstack(filtered_points_list)
#
# # Function to fit an ellipse
# def ellipse_residuals(params, x, y):
#     xc, yc, a, b, theta = params
#     cos_t = np.cos(theta)
#     sin_t = np.sin(theta)
#     x_new = cos_t * (x - xc) + sin_t * (y - yc)
#     y_new = -sin_t * (x - xc) + cos_t * (y - yc)
#     return ((x_new / a) ** 2 + (y_new / b) ** 2) - 1
#
# def fit_ellipse(x, y):
#     x_m = np.mean(x)
#     y_m = np.mean(y)
#     initial_guess = [x_m, y_m, np.std(x), np.std(y), 0]
#     result = least_squares(ellipse_residuals, initial_guess, args=(x, y))
#     return result.x
#
# # Fit ellipses for both sets of filtered points
# ellipse_params = []
# for points_set in filtered_points_list:
#     x = points_set[:, 0]
#     y = points_set[:, 1]
#     params = fit_ellipse(x, y)
#     ellipse_params.append(params)
#
# # Plot the result
# plt.scatter(filtered_points[:, 0], filtered_points[:, 1], s=10, color='black', label='Filtered Points')
#
# # Plot the fitted ellipses
# for params in ellipse_params:
#     xc, yc, a, b, theta = params
#     t = np.linspace(0, 2 * np.pi, 100)
#     Ell = np.array([a * np.cos(t), b * np.sin(t)])
#     R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
#     Ell_rot = np.dot(R, Ell)
#     plt.plot(xc + Ell_rot[0, :], yc + Ell_rot[1, :], label=f'Ellipse: a={a:.2f}, b={b:.2f}, θ={theta:.2f}')
#
# plt.legend()
# plt.title('Fitted Ellipses')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
#
# # Print the ellipse parameters
# for i, params in enumerate(ellipse_params):
#     xc, yc, a, b, theta = params
#     print(f"Ellipse {i + 1}: Center=({xc:.2f}, {yc:.2f}), a={a:.2f}, b={b:.2f}, θ={theta:.2f} radians")

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
# import pandas as pd
# from scipy.optimize import least_squares
#
# # Load your points (x, y) here
# df = pd.read_excel("RF_free_data1.xlsx", sheet_name="RF_combined_df")
# df.drop_duplicates(subset=['X', 'Y'], keep='first', inplace=True)
# df = df[['X', 'Y']]
# x = np.array(df['X'])
# y = np.array(df['Y'])
# points = np.column_stack((x, y))
#
# # Apply DBSCAN to find clusters
# dbscan = DBSCAN(eps=1e6, min_samples=10)
# labels = dbscan.fit_predict(points)
#
# # Check if any clusters were found
# if len(set(labels)) <= 1:  # Only noise or a single cluster found
#     raise ValueError("DBSCAN did not find any clusters. Adjust 'eps' and 'min_samples' parameters.")
#
# # Calculate the centers of the detected clusters
# unique_labels = set(labels)
# centers = []
# for label in unique_labels:
#     if label != -1:  # Exclude noise
#         cluster_points = points[labels == label]
#         center = cluster_points.mean(axis=0)
#         centers.append(center)
#
# # Calculate distances from each point to the nearest center
# filtered_points_list = []
# max_distance_points = None
# max_distance = -np.inf
#
# for center in centers:
#     distances = np.linalg.norm(points - center, axis=1)
#     threshold = np.median(distances)
#     filtered_points = points[distances < threshold]
#     filtered_points_list.append(filtered_points)
#
#     # Check if this is the outermost ellipse based on max distance
#     current_max_distance = np.max(distances)
#     if current_max_distance > max_distance:
#         max_distance = current_max_distance
#         max_distance_points = filtered_points
#
# # Ensure that filtered_points_list is a list of 2D arrays
# if not filtered_points_list:
#     raise ValueError("No points found within the specified thresholds.")
#
# # Now we only keep the outermost ellipse points
# filtered_points = max_distance_points
#
# # Function to fit an ellipse
# def ellipse_residuals(params, x, y):
#     xc, yc, a, b, theta = params
#     cos_t = np.cos(theta)
#     sin_t = np.sin(theta)
#     x_new = cos_t * (x - xc) + sin_t * (y - yc)
#     y_new = -sin_t * (x - xc) + cos_t * (y - yc)
#     return ((x_new / a) ** 2 + (y_new / b) ** 2) - 1
#
# def fit_ellipse(x, y):
#     x_m = np.mean(x)
#     y_m = np.mean(y)
#     initial_guess = [x_m, y_m, np.std(x), np.std(y), 0]
#     result = least_squares(ellipse_residuals, initial_guess, args=(x, y))
#     return result.x
#
# # Fit ellipse for the outermost set of filtered points
# x = filtered_points[:, 0]
# y = filtered_points[:, 1]
# ellipse_params = fit_ellipse(x, y)
#
# # Plot the result
# plt.scatter(points[:, 0], points[:, 1], s=10, color='grey', alpha=0.3, label='Original Points')
# plt.scatter(filtered_points[:, 0], filtered_points[:, 1], s=10, color='black', label='Filtered Points (Outermost Ellipse)')
#
# # Plot the fitted ellipse
# xc, yc, a, b, theta = ellipse_params
# t = np.linspace(0, 2 * np.pi, 100)
# Ell = np.array([a * np.cos(t), b * np.sin(t)])
# R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
# Ell_rot = np.dot(R, Ell)
# plt.plot(xc + Ell_rot[0, :], yc + Ell_rot[1, :], label=f'Fitted Ellipse: a={a:.2f}, b={b:.2f}, θ={theta:.2f}')
#
# plt.legend()
# plt.title('Outermost Fitted Ellipse')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
#
# # Print the ellipse parameters
# xc, yc, a, b, theta = ellipse_params
# print(f"Outermost Ellipse: Center=({xc:.2f}, {yc:.2f}), a={a:.2f}, b={b:.2f}, θ={theta:.2f} radians")

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
# import pandas as pd
# from scipy.optimize import least_squares
# # RF_combined_df
# # 加载点 (x, y)
# df = pd.read_excel("RF_free_data1.xlsx", sheet_name="RF_combined_df")
# df.drop_duplicates(subset=['X', 'Y'], keep='first', inplace=True)
# df = df[['X', 'Y']]
# x = np.array(df['X'])
# y = np.array(df['Y'])
# points = np.column_stack((x, y))
#
# # 应用DBSCAN找到聚类
# dbscan = DBSCAN(eps=1e6, min_samples=10)
# labels = dbscan.fit_predict(points)
#
# # 检查是否找到任何聚类
# if len(set(labels)) <= 1:  # 仅噪声或单个聚类
#     raise ValueError("DBSCAN未找到任何聚类。调整'eps'和'min_samples'参数。")
#
# # 计算检测到的聚类的中心
# unique_labels = set(labels)
# centers = []
# for label in unique_labels:
#     if label != -1:  # 排除噪声
#         cluster_points = points[labels == label]
#         center = cluster_points.mean(axis=0)
#         centers.append(center)
#
# # 计算每个点到最近中心的距离
# max_distance_points = None
# max_distance = -np.inf
#
# for center in centers:
#     distances = np.linalg.norm(points - center, axis=1)
#     threshold = np.percentile(distances, 90)  # 使用90百分位作为阈值
#     filtered_points = points[distances > threshold]  # 仅保留距离超过阈值的点
#
#     # 检查是否为最外层椭圆
#     current_max_distance = np.max(distances)
#     if current_max_distance > max_distance:
#         max_distance = current_max_distance
#         max_distance_points = filtered_points
#
# # 确保filtered_points_list是一个2D数组的列表
# if max_distance_points is None:
#     raise ValueError("在指定阈值内未找到任何点。")
#
# # 仅保留最外层椭圆点
# filtered_points = max_distance_points
#
# # 椭圆残差计算
# def ellipse_residuals(params, x, y):
#     xc, yc, a, b, theta = params
#     cos_t = np.cos(theta)
#     sin_t = np.sin(theta)
#     x_new = cos_t * (x - xc) + sin_t * (y - yc)
#     y_new = -sin_t * (x - xc) + cos_t * (y - yc)
#     return ((x_new / a) ** 2 + (y_new / b) ** 2) - 1
#
# # 拟合椭圆函数
# def fit_ellipse(x, y):
#     x_m = np.mean(x)
#     y_m = np.mean(y)
#     initial_guess = [x_m, y_m, np.std(x), np.std(y), 0]
#     result = least_squares(ellipse_residuals, initial_guess, args=(x, y))
#     return result.x
#
# # 拟合最外层过滤点集的椭圆
# x = filtered_points[:, 0]
# y = filtered_points[:, 1]
# ellipse_params = fit_ellipse(x, y)
#
# # 绘图
# plt.scatter(points[:, 0], points[:, 1], s=10, color='grey', alpha=0.3, label='原始点')
# plt.scatter(filtered_points[:, 0], filtered_points[:, 1], s=10, color='black', label='过滤点（最外层椭圆）')
#
# # 绘制拟合的椭圆
# xc, yc, a, b, theta = ellipse_params
# t = np.linspace(0, 2 * np.pi, 100)
# Ell = np.array([a * np.cos(t), b * np.sin(t)])
# R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
# Ell_rot = np.dot(R, Ell)
# plt.plot(xc + Ell_rot[0, :], yc + Ell_rot[1, :], label=f'拟合椭圆: a={a:.2f}, b={b:.2f}, θ={theta:.2f}')
#
# plt.legend()
# plt.title('最外层拟合椭圆')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
#
# # 打印椭圆参数
# xc, yc, a, b, theta = ellipse_params
# print(f"最外层椭圆: 中心=({xc:.2f}, {yc:.2f}), a={a:.2f}, b={b:.2f}, θ={theta:.2f} radians")

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.covariance import EllipticEnvelope
# from scipy.spatial import distance
#
# # 加载数据并去重
# df = pd.read_excel("RF_free_data1.xlsx", sheet_name="RF_max")
# df.drop_duplicates(subset=['X', 'Y'], keep='first', inplace=True)
# df = df[['X', 'Y']]
# x = np.array(df['X'])
# y = np.array(df['Y'])
# points = np.column_stack((x, y))
#
# # 初步拟合椭圆
# elliptic_env = EllipticEnvelope(support_fraction=1.0)
# elliptic_env.fit(points)
# center = elliptic_env.location_
#
# # 计算每个点相对于中心点的方向角和距离
# angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
# distances = distance.cdist(points, [center])
#
# # 细分角度范围，每个范围保留一个最远的点
# num_bins = 100  # 细分角度的数量，可以调整这个值
# angle_bins = np.linspace(-np.pi, np.pi, num_bins)
# indices = np.digitize(angles, angle_bins)
#
# # 创建一个字典来存储每个方向上最远的点
# max_distance_points = {}
#
# for i in range(len(points)):
#     bin_idx = indices[i]
#     if bin_idx not in max_distance_points or distances[i] > max_distance_points[bin_idx][0]:
#         max_distance_points[bin_idx] = (distances[i], points[i])
#
# # 提取过滤后的点集
# filtered_points = np.array([point for _, point in max_distance_points.values()])
#
# # 再次拟合过滤后的点集
# elliptic_env_filtered = EllipticEnvelope(support_fraction=1.0)
# elliptic_env_filtered.fit(filtered_points)
# filtered_center = elliptic_env_filtered.location_
#
# # 绘制初步拟合和过滤后的拟合结果
# plt.figure(figsize=(12, 6))
#
# plt.subplot(122)
# plt.scatter(filtered_points[:, 0], filtered_points[:, 1], s=4)
# plt.scatter(filtered_center[0], filtered_center[1], color='red')
# plt.title("Filtered Fit")
#
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope
from scipy.spatial import distance
from scipy.optimize import least_squares

def Curve_fitting(RF_max, RF_min):
    def Fit_ellipse_function(filtered_points, points):
        # 椭圆残差计算
        def ellipse_residuals(params, x, y):
            xc, yc, a, b, theta = params
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            x_new = cos_t * (x - xc) + sin_t * (y - yc)
            y_new = -sin_t * (x - xc) + cos_t * (y - yc)
            return ((x_new / a) ** 2 + (y_new / b) ** 2) - 1

        # 拟合椭圆函数
        def fit_ellipse(x, y):
            x_m = np.mean(x)
            y_m = np.mean(y)
            initial_guess = [x_m, y_m, np.std(x), np.std(y), 0]
            result = least_squares(ellipse_residuals, initial_guess, args=(x, y))
            return result.x

        # 从过滤点集拟合椭圆
        x = filtered_points[:, 0]
        y = filtered_points[:, 1]
        ellipse_params = fit_ellipse(x, y)

        # # 绘图
        # plt.scatter(points[:, 0], points[:, 1], s=10, color='grey', alpha=0.3, label='Original Points')
        # plt.scatter(filtered_points[:, 0], filtered_points[:, 1], s=10, color='black',
        #             label='Filtered Points (Outermost Ellipse)')
        #
        # # 绘制拟合的椭圆
        # xc, yc, a, b, theta = ellipse_params
        # t = np.linspace(0, 2 * np.pi, 100)
        # Ell = np.array([a * np.cos(t), b * np.sin(t)])
        # R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        # Ell_rot = np.dot(R, Ell)
        # plt.plot(xc + Ell_rot[0, :], yc + Ell_rot[1, :], label=f'Fitted Ellipse: a={a:.2f}, b={b:.2f}, θ={theta:.2f}')
        #
        # plt.legend()
        # plt.title('Outermost Fitted Ellipse')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.show()
        #
        # # 打印椭圆参数
        # xc, yc, a, b, theta = ellipse_params
        # print(f"Filtered Ellipse: Center=({xc:.2f}, {yc:.2f}), a={a:.2f}, b={b:.2f}, θ={theta:.2f} radians")

        return ellipse_params

    def data_processing(data, flag):
        xy_coordinates = data[:, :2]
        unique_xy = np.unique(xy_coordinates, axis=0)
        x = unique_xy[:, 0]
        y = unique_xy[:, 1]

        points = np.column_stack((x, y))
        # 检测包含NaN值的行
        nan_mask = np.isnan(points).any(axis=1)
        points = points[~nan_mask]

        # 初步拟合椭圆
        elliptic_env = EllipticEnvelope(support_fraction=1.0)
        elliptic_env.fit(points)
        center = elliptic_env.location_

        # 计算每个点相对于中心点的方向角和距离
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        distances = distance.cdist(points, [center])

        # 细分角度范围，每个范围保留一个最近的点
        num_bins = 100  # 细分角度的数量，可以调整这个值
        angle_bins = np.linspace(-np.pi, np.pi, num_bins)
        indices = np.digitize(angles, angle_bins)

        # 创建一个字典来存储每个方向上最近/最大的点
        distance_points = {}

        if flag:
            for i in range(len(points)):
                bin_idx = indices[i]
                if bin_idx not in distance_points or distances[i] > distance_points[bin_idx][0]:
                    distance_points[bin_idx] = (distances[i], points[i])
        else:
            for i in range(len(points)):
                bin_idx = indices[i]
                if bin_idx not in distance_points or distances[i] < distance_points[bin_idx][0]:
                    distance_points[bin_idx] = (distances[i], points[i])

        # 提取过滤后的点集
        filtered_points = np.array([point for _, point in distance_points.values()])

        # 再次拟合过滤后的点集
        elliptic_env_filtered = EllipticEnvelope(support_fraction=1.0)
        elliptic_env_filtered.fit(filtered_points)
        ellipse_params = Fit_ellipse_function(filtered_points, points)

        return ellipse_params

    ellipse_params_c = data_processing(np.array(RF_max), 1)
    ellipse_params_t = data_processing(np.array(RF_min), 0)

    return np.array([ellipse_params_c, ellipse_params_t])

# RF_max = pd.read_excel("RF_free_data2.xlsx", sheet_name="RF_max")
# RF_min = pd.read_excel("RF_free_data2.xlsx", sheet_name="RF_min")
# Curve_fitting(RF_max, RF_min)


