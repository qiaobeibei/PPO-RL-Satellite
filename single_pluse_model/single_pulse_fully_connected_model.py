# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, random_split
# from torch.nn.utils.rnn import pad_sequence
# import pandas as pd
# import matplotlib.pyplot as plt
# from tqdm import tqdm
#
# # 定义自定义数据集类
# class CustomDataset(Dataset):
#     def __init__(self, input_df, result_df):
#         self.input_df = input_df
#         self.result_df = result_df
#
#     def __len__(self):
#         return len(self.input_df)
#
#     def __getitem__(self, idx):
#         input_data = self.input_df.iloc[idx].values.astype('float32')
#         result_data = self.result_df.iloc[idx].values.astype('float32')
#         return torch.tensor(input_data), torch.tensor(result_data)
#
# # 自定义collate_fn函数
# def collate_fn(batch):
#     inputs, targets = zip(*batch)
#     inputs = torch.stack(inputs, 0)
#     target_lengths = [len(target) for target in targets]
#     # max_length = max(target_lengths)
#
#     padded_targets = []
#     for target in targets:
#         if len(target) < max_length:
#             padding = torch.zeros((max_length - len(target), 3))
#             padded_target = torch.cat([target, padding], dim=0)
#         else:
#             padded_target = target
#         padded_targets.append(padded_target)
#
#     targets = torch.stack(padded_targets, 0)
#     return inputs, targets, torch.tensor(target_lengths), max_length
#
# # 读取输入数据的CSV文件
# input_data_file = 'all_input.csv'
# input_df = pd.read_csv(input_data_file)
# print("输入数据:")
# print(input_df.head())
#
# # 读取结果数据的CSV文件
# result_data_file = 'output_data.csv'
# output_df = pd.read_csv(result_data_file)
# print("结果数据:")
# print(output_df.head())
#
# # 创建数据集
# dataset = CustomDataset(input_df, output_df)
# num_samples = len(dataset)
#
# # 将数据分为训练集和验证集
# train_size = int(0.85 * num_samples)
# val_size = num_samples - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# print(f'训练集大小: {train_size}')
# print(f'验证集大小: {val_size}')
#
# # 创建数据加载器
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#
# # 设计改进后的神经网络
# class NeuralNetwork(nn.Module):
#     def __init__(self, input_dim):
#         super(NeuralNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 256)
#         self.fc3 = nn.Linear(256, 64)
#         self.output_layer = nn.Linear(64, 10)  # 这里的 3 表示每个点的 3 个坐标值
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = self.output_layer(x)
#         return x
#
# input_dim = 5  # 输入维度应该是 5 (a, e, i, omega, fuel)
# model = NeuralNetwork(input_dim)
#
# initial_params = {name: param.clone() for name, param in model.state_dict().items()}
#
# # 设置损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.1)
#
# # 查看 train_loader 的输出
# for batch_idx, (inputs, targets) in enumerate(train_loader):
#     print(f"Batch {batch_idx + 1}:")
#     print(f"Inputs shape: {inputs.shape}")
#     print(f"Targets shape: {targets.shape}")
#     print(f"Inputs: {inputs}")
#     print(f"Targets: {targets}")
#     break  # 只查看第一个批量
#
# # 训练模型
# num_epochs = 400
# train_losses = []
# val_losses = []
#
# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0
#     for X_batch, y_batch in train_loader:
#         optimizer.zero_grad()
#         y_pred = model(X_batch)
#         loss = criterion(y_pred, y_batch)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item() * X_batch.size(0)
#     train_loss /= train_size
#     train_losses.append(train_loss)
#
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for X_batch, y_batch in val_loader:
#             y_pred = model(X_batch)
#             loss = criterion(y_pred, y_batch)
#             val_loss += loss.item() * X_batch.size(0)
#     val_loss /= val_size
#     val_losses.append(val_loss)
#
#     # changed = False
#     # for name, param in model.state_dict().items():
#     #     if not torch.equal(param, initial_params[name]):
#     #         print(f"Parameter '{name}' has changed.")
#     #         changed = True
#     #
#     # if not changed:
#     #     print("No parameters have changed.")
#     # else:
#     #     print("Some parameters have changed.")
#
#     print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
#
# # 绘制训练和验证损失
# plt.plot(train_losses, label='train loss')
# plt.plot(val_losses, label='val loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# # 测试模型
# test_input = None
# model.eval()
# with torch.no_grad():
#     predicted_output = model(test_input, max_length)
# print(predicted_output)


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
#
# # 读取数据
# input_data = pd.read_csv('all_input.csv')
# output_data = pd.read_csv('output_data.csv')
#
# # 数据标准化
# input_scaler = StandardScaler()
# output_scaler = StandardScaler()
#
# inputs = input_scaler.fit_transform(input_data.values)
# outputs = output_scaler.fit_transform(output_data.values)
#
# # 转换为张量
# inputs = torch.tensor(inputs, dtype=torch.float32)
# outputs = torch.tensor(outputs, dtype=torch.float32)
#
#
# # 自定义数据集
# class CustomDataset(Dataset):
#     def __init__(self, inputs, outputs):
#         self.inputs = inputs
#         self.outputs = outputs
#
#     def __len__(self):
#         return len(self.inputs)
#
#     def __getitem__(self, idx):
#         return self.inputs[idx], self.outputs[idx]
#
#
# # 创建数据集和数据加载器
# dataset = CustomDataset(inputs, outputs)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
#
# # 定义神经网络模型
# class ImprovedNN(nn.Module):
#     def __init__(self):
#         super(ImprovedNN, self).__init__()
#         self.fc1 = nn.Linear(5, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 64)
#         self.fc4 = nn.Linear(64, 10)
#         self.dropout = nn.Dropout(0.2)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = torch.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = torch.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x
#
#
# # 初始化模型
# model = ImprovedNN()
#
# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#
# # 训练模型
# num_epochs = 500
# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss = 0
#     for batch_inputs, batch_outputs in dataloader:
#         optimizer.zero_grad()
#         predictions = model(batch_inputs)
#         loss = criterion(predictions, batch_outputs)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#
#     scheduler.step()
#
#     if (epoch + 1) % 10 == 0:
#         print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")
#
# # 验证模型（示例）
# model.eval()
# with torch.no_grad():
#     sample_input = inputs[0].unsqueeze(0)  # 获取一个输入样本
#     predicted_output = model(sample_input)
#     predicted_output = output_scaler.inverse_transform(predicted_output.numpy())
#     actual_output = output_scaler.inverse_transform(outputs[0].unsqueeze(0).numpy())
#     print(f"Predicted Output: {predicted_output}")
#     print(f"Actual Output: {actual_output}")


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import os
from model import ImprovedNN, CustomDataset

# 读取数据
input_data = pd.read_csv('all_input.csv')
output_data = pd.read_csv('output_data.csv')

# 数据标准化
input_scaler = StandardScaler()
output_scaler = StandardScaler()

inputs = input_scaler.fit_transform(input_data.values)
outputs = output_scaler.fit_transform(output_data.values)

# 转换为张量
inputs = torch.tensor(inputs, dtype=torch.float32)
outputs = torch.tensor(outputs, dtype=torch.float32)


# 创建数据集
dataset = CustomDataset(inputs, outputs)

# 分割数据集，80% 训练集，20% 测试集
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 初始化模型
save_path = './MLPNet.pth'
# 是否使用预训练模型
pre_train = False
model = ImprovedNN()
if pre_train:
    assert os.path.exists(save_path), "file: '{}' dose not exist.".format(save_path)
    model.load_state_dict(torch.load(save_path))

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 训练模型
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_inputs, batch_outputs in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_inputs)
        loss = criterion(predictions, batch_outputs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader)}")

# 在测试集上验证模型
model.eval()
test_loss = 0
with torch.no_grad():
    for batch_inputs, batch_outputs in test_loader:
        predictions = model(batch_inputs)
        loss = criterion(predictions, batch_outputs)
        test_loss += loss.item()

    print(f"Test Loss: {test_loss / len(test_loader)}")

# 使用测试集中的一个样本进行预测
with torch.no_grad():
    sample_input = test_dataset[0][0].unsqueeze(0)  # 获取一个测试输入样本
    predicted_output = model(sample_input)
    predicted_output = output_scaler.inverse_transform(predicted_output.numpy())
    actual_output = output_scaler.inverse_transform(test_dataset[0][1].unsqueeze(0).numpy())
    print(f"Predicted Output: {predicted_output}")
    print(f"Actual Output: {actual_output}")

model_name = 'MLPNet'  + '.pth'
chkpt_file = os.path.join(os.getcwd(), model_name)
torch.save(model.state_dict(), chkpt_file)
