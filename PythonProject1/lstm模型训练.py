import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_lstm_model(base_path):
    # 设置文件路径
    # base_path = "C:/Users/刘欣逸/Desktop/毕设/"
    train_file = base_path + "train_sequences.npy"
    val_file = base_path + "val_sequences.npy"
    test_file = base_path + "test_sequences.npy"

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    train_data = np.load(train_file)
    val_data = np.load(val_file)
    test_data = np.load(test_file)

    # 定义 LSTM 输入、输出
    seq_length = train_data.shape[1]  # 20
    input_size = train_data.shape[2]  # 3（纬度、经度、高度）


    # 划分输入 (X) 和 目标 (Y)
    def prepare_data(data):
        X = data[:, :-1, :]  # 取前 (seq_length-1) 作为输入
        Y = data[:, -1, :]  # 取最后一个时间步作为预测目标
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


    X_train, Y_train = prepare_data(train_data)
    X_val, Y_val = prepare_data(val_data)
    X_test, Y_test = prepare_data(test_data)

    # 转换为 DataLoader
    batch_size = 64
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False, pin_memory=True)


    # 定义 LSTM 模型
    class TrajectoryLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(TrajectoryLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            # 在forward方法中自动初始化h0和c0
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

            # LSTM 前向传播
            out, _ = self.lstm(x, (h0, c0))

            # 取最后一个时间步的输出
            out = self.fc(out[:, -1, :])
            return out


    # 初始化模型
    hidden_size = 32
    num_layers = 1
    output_size = input_size

    model = TrajectoryLSTM(input_size, hidden_size, num_layers, output_size).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 使用学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 训练模型
    num_epochs = 50

    def training_generator():
        best_val_loss = float("inf")
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

                optimizer.zero_grad()

                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, Y_batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # 生成进度事件
            yield {
                "type": "progress",
                "epoch": epoch + 1,
                "total_epochs": 50,
                "train_loss": train_loss,
                "val_loss": val_loss
            }

            # 保存最优模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), base_path + "best_lstm_model.pth")

            # 更新学习率
            scheduler.step()

        print("训练完成，最佳模型已保存！")

        # 测试模型
        model.load_state_dict(torch.load(base_path + "best_lstm_model.pth"))
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        yield {
            "type": "complete",
            "test_loss": test_loss
        }
        print(f"测试集损失: {test_loss:.6f}")

    return training_generator()

# import torch
# import numpy as np
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
#
# # 设置文件路径
# base_path = "C:/Users/刘欣逸/Desktop/毕设/"
# train_file = base_path + "train_sequences.npy"
# val_file = base_path + "val_sequences.npy"
# test_file = base_path + "test_sequences.npy"
#
# # 设备设置
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 加载数据
# train_data = np.load(train_file)
# val_data = np.load(val_file)
# test_data = np.load(test_file)
#
# # 定义 LSTM 输入、输出
# seq_length = train_data.shape[1]  # 20
# input_size = train_data.shape[2]  # 3（纬度、经度、高度）
#
#
# # 划分输入 (X) 和 目标 (Y)
# def prepare_data(data):
#     X = data[:, :-1, :]  # 取前 (seq_length-1) 作为输入
#     Y = data[:, -1, :]  # 取最后一个时间步作为预测目标
#     return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
#
#
# X_train, Y_train = prepare_data(train_data)
# X_val, Y_val = prepare_data(val_data)
# X_test, Y_test = prepare_data(test_data)
#
# # 转换为 DataLoader
# batch_size = 64
# train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True, pin_memory=True)
# val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False, pin_memory=True)
# test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False, pin_memory=True)
#
#
# # 定义 LSTM 模型
# class TrajectoryLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(TrajectoryLSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         # 在forward方法中自动初始化h0和c0
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
#
#         # LSTM 前向传播
#         out, _ = self.lstm(x, (h0, c0))
#
#         # 取最后一个时间步的输出
#         out = self.fc(out[:, -1, :])
#         return out
#
#
# # 初始化模型
# hidden_size = 32
# num_layers = 1
# output_size = input_size
#
# model = TrajectoryLSTM(input_size, hidden_size, num_layers, output_size).to(device)
#
# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 使用学习率调度器
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
#
# # 训练模型
# num_epochs = 50
# best_val_loss = float("inf")
#
# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0
#     for X_batch, Y_batch in train_loader:
#         X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
#
#         optimizer.zero_grad()
#
#         outputs = model(X_batch)
#         loss = criterion(outputs, Y_batch)
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item()
#
#     train_loss /= len(train_loader)
#
#     # 验证
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for X_batch, Y_batch in val_loader:
#             X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
#             outputs = model(X_batch)
#             loss = criterion(outputs, Y_batch)
#             val_loss += loss.item()
#
#     val_loss /= len(val_loader)
#
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
#
#     # 保存最优模型
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(model.state_dict(), base_path + "best_lstm_model.pth")
#
#     # 更新学习率
#     scheduler.step()
#
# print("训练完成，最佳模型已保存！")
#
# # 测试模型
# model.load_state_dict(torch.load(base_path + "best_lstm_model.pth"))
# model.eval()
# test_loss = 0
# with torch.no_grad():
#     for X_batch, Y_batch in test_loader:
#         X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
#         outputs = model(X_batch)
#         loss = criterion(outputs, Y_batch)
#         test_loss += loss.item()
#
# test_loss /= len(test_loader)
# print(f"测试集损失: {test_loss:.6f}")
