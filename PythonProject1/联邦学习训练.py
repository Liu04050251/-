import numpy as np
import uuid  # 添加uuid库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import syft as sy
from io import BytesIO
import base64
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def federated_train(base_path, num_clients, as_stream=False):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # 1. 创建 PySyft 虚拟客户端
    # ------------------------------
    #num_clients = 5
    clients = [sy.Domain(name=f"client_{i}") for i in range(num_clients)]

    # ------------------------------
    # 2. 加载数据
    # ------------------------------
    #base_path = "C:/Users/刘欣逸/Desktop/毕设/"
    train_data = np.load(base_path + "train_sequences.npy")
    val_data = np.load(base_path + "val_sequences.npy")
    test_data = np.load(base_path + "test_sequences.npy")

    train_user_ids = np.load(base_path + "train_user_ids.npy")
    val_user_ids = np.load(base_path + "val_user_ids.npy")
    test_user_ids = np.load(base_path + "test_user_ids.npy")

    unique_users = np.unique(train_user_ids)

    # ------------------------------
    # 3. 分配数据给客户端
    # ------------------------------
    client_datasets = {client.name: [] for client in clients}

    for idx, user_id in enumerate(unique_users):
        user_train_data = train_data[train_user_ids == user_id]
        client = clients[idx % num_clients]
        client_datasets[client.name].append(user_train_data)

    # ------------------------------
    # 4. 数据预处理
    # ------------------------------
    def prepare_data(data):
        assert len(data.shape) == 3, "输入数据应为 3D：样本数 × 时间步 × 特征数"
        X = data[:, :-1, :]  # 输入为前19步
        Y = data[:, -1, :]   # 标签为第20步
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

    client_loaders = {}
    batch_size = 32

    for client in clients:
        client_name = client.name
        if client_name in client_datasets and client_datasets[client_name]:
            client_data = np.concatenate(client_datasets[client_name], axis=0)
            X, Y = prepare_data(client_data)
            client_loaders[client] = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)

    X_val, Y_val = prepare_data(val_data)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

    # ------------------------------
    # 5. 模型定义
    # ------------------------------
    class TrajectoryLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(TrajectoryLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

    input_size = 3
    hidden_size = 32
    num_layers = 1
    output_size = 3
    global_model = TrajectoryLSTM(input_size, hidden_size, num_layers, output_size).to(device)

    # ------------------------------
    # 6. 联邦训练
    # ------------------------------
    val_losses = []

    # 训练过程监控数据
    result = {
        "client_losses": {client.name: [] for client in clients},
        "val_metrics": [],
        "loss_plot": ""
    }

    def train_local(client, model, data_loader, epochs=5, lr=0.001):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        client_loss = []

        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (X_batch, Y_batch) in enumerate(data_loader):
                if epoch == 0 and batch_idx == 0:
                    # 生成debug_info事件
                    debug_info = {
                        "type": "debug_info",
                        "client": client.name,
                        "input_shape": list(X_batch.shape),
                        "label_shape": list(Y_batch.shape)
                    }
                    yield debug_info
                    print(f"[DEBUG] 客户端 {client.name} 输入维度: {X_batch.shape}, 标签维度: {Y_batch.shape}")
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            epoch_loss = total_loss / len(data_loader)
            client_loss.append(epoch_loss)

            print(f"客户端 {client.name} - 轮次 {epoch + 1}, 训练损失: {total_loss / len(data_loader):.6f}")

            # 如果是流式输出，返回训练进度
            if as_stream:
                yield {
                    "type": "client_progress",
                    "client": client.name,
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "loss": epoch_loss
                }

        # 记录客户端训练详情
        result["client_losses"][client.name].extend(client_loss)
        return model.state_dict()

    def federated_training(global_model, clients, client_loaders, val_loader, rounds=10):
        global_model.train()

        for r in range(rounds):
            if as_stream:
                yield {"type": "round_start", "round": r + 1, "total_rounds": rounds}

            print(f"\n--- 联邦训练轮次 {r + 1} ---")
            local_weights = []
            total_samples = 0

            for client in clients:
                if client in client_loaders:
                    if as_stream:
                        yield {
                            "type": "client_start",
                            "client": client.name,
                            "samples": len(client_loaders[client].dataset),
                            "round": r + 1
                        }
                    local_model = TrajectoryLSTM(input_size, hidden_size, num_layers, output_size).to(device)
                    local_model.load_state_dict(global_model.state_dict())
                    # 处理流式训练
                    if as_stream:
                        for progress in train_local(client, local_model, client_loaders[client]):
                            yield progress
                        local_weight = local_model.state_dict()
                    else:
                        local_weight = train_local(client, local_model, client_loaders[client])

                    num_samples = len(client_loaders[client].dataset)
                    total_samples += num_samples
                    local_weights.append((num_samples, local_weight))

            avg_weights = OrderedDict()
            for key in global_model.state_dict().keys():
                weighted_sum = sum(w[key] * n for n, w in local_weights)
                avg_weights[key] = weighted_sum / total_samples

            global_model.load_state_dict(avg_weights)
            print(f"联邦轮次 {r + 1} 结束，全局模型已更新")

            # 验证集评估
            val_metrics = test_model(global_model, val_loader, dataset_type="验证集", record_loss=True)

            if as_stream:
                yield {
                    "type": "validation",
                    "round": r + 1,
                    "mse": val_metrics["mse"],
                    "mae": val_metrics["mae"],
                    "r2": val_metrics["r2"]
                }
    # ------------------------------
    # 7. 模型测试与评估
    # ------------------------------
    def test_model(model, test_loader, dataset_type="测试集", record_loss=False):
        model.eval()
        criterion = nn.MSELoss()
        test_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                test_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(Y_batch.cpu().numpy())

        test_loss /= len(test_loader)
        print(f"{dataset_type} MSE损失: {test_loss:.6f}")

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        mae = mean_absolute_error(targets, preds)
        r2 = r2_score(targets, preds)

        print(f"{dataset_type} MAE: {mae:.6f}")
        print(f"{dataset_type} R² Score: {r2:.6f}")

        # 新增指标记录
        metrics = {
            "round": len(result["val_metrics"]) + 1,
            "mse": test_loss,
            "mae": mae,
            "r2": r2
        }
        result["val_metrics"].append(metrics)

        if record_loss:
            val_losses.append(test_loss)

        return metrics

    # ------------------------------
    # 8. 执行训练与测试
    # ------------------------------
    if as_stream:
        # 流式训练模式
        for progress in federated_training(global_model, clients, client_loaders, val_loader, rounds=10):
            yield progress
    else:
        # 传统训练模式
        federated_training(global_model, clients, client_loaders, val_loader, rounds=10)

    X_test, Y_test = prepare_data(test_data)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)
    test_model(global_model, test_loader, dataset_type="测试集")

    # ------------------------------
    # 9. 可视化验证集损失变化
    # ------------------------------
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o')
    plt.title("联邦训练中验证集损失变化")
    plt.xlabel("联邦轮次")
    plt.ylabel("验证集 MSE Loss")
    plt.grid(True)
    #plt.show()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    result["loss_plot"] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    if not as_stream:
        return result
    else:
        # 流式模式下最后返回完整结果
        yield {
            "type": "final_result",
            "result": {
                "loss_plot": base64.b64encode(buf.getvalue()).decode('utf-8'),
                "val_metrics": result["val_metrics"]
            }
        }

#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import syft as sy
# from collections import OrderedDict
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error, r2_score
# import matplotlib
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False
#
# # 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # ------------------------------
# # 1. 创建 PySyft 虚拟客户端
# # ------------------------------
# num_clients = 5
# clients = [sy.Domain(name=f"client_{i}") for i in range(num_clients)]
#
# # ------------------------------
# # 2. 加载数据
# # ------------------------------
# base_path = "C:/Users/刘欣逸/Desktop/毕设/"
# train_data = np.load(base_path + "train_sequences.npy")
# val_data = np.load(base_path + "val_sequences.npy")
# test_data = np.load(base_path + "test_sequences.npy")
#
# train_user_ids = np.load(base_path + "train_user_ids.npy")
# val_user_ids = np.load(base_path + "val_user_ids.npy")
# test_user_ids = np.load(base_path + "test_user_ids.npy")
#
# unique_users = np.unique(train_user_ids)
#
# # ------------------------------
# # 3. 分配数据给客户端
# # ------------------------------
# client_datasets = {client.name: [] for client in clients}
#
# for idx, user_id in enumerate(unique_users):
#     user_train_data = train_data[train_user_ids == user_id]
#     client = clients[idx % num_clients]
#     client_datasets[client.name].append(user_train_data)
#
# # ------------------------------
# # 4. 数据预处理
# # ------------------------------
# def prepare_data(data):
#     assert len(data.shape) == 3, "输入数据应为 3D：样本数 × 时间步 × 特征数"
#     X = data[:, :-1, :]  # 输入为前19步
#     Y = data[:, -1, :]   # 标签为第20步
#     return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
#
# client_loaders = {}
# batch_size = 32
#
# for client in clients:
#     client_name = client.name
#     if client_name in client_datasets and client_datasets[client_name]:
#         client_data = np.concatenate(client_datasets[client_name], axis=0)
#         X, Y = prepare_data(client_data)
#         client_loaders[client] = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)
#
# X_val, Y_val = prepare_data(val_data)
# val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)
#
# # ------------------------------
# # 5. 模型定义
# # ------------------------------
# class TrajectoryLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(TrajectoryLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])
#         return out
#
# input_size = 3
# hidden_size = 32
# num_layers = 1
# output_size = 3
# global_model = TrajectoryLSTM(input_size, hidden_size, num_layers, output_size).to(device)
#
# # ------------------------------
# # 6. 联邦训练
# # ------------------------------
# val_losses = []
#
# def train_local(client, model, data_loader, epochs=5, lr=0.001):
#     model.train()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.MSELoss()
#
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch_idx, (X_batch, Y_batch) in enumerate(data_loader):
#             if epoch == 0 and batch_idx == 0:
#                 print(f"[DEBUG] 客户端 {client.name} 输入维度: {X_batch.shape}, 标签维度: {Y_batch.shape}")
#             X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
#             optimizer.zero_grad()
#             outputs = model(X_batch)
#             loss = criterion(outputs, Y_batch)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#
#         print(f"客户端 {client.name} - 轮次 {epoch + 1}, 训练损失: {total_loss / len(data_loader):.6f}")
#
#     return model.state_dict()
#
# def federated_training(global_model, clients, client_loaders, val_loader, rounds=10):
#     global_model.train()
#
#     for r in range(rounds):
#         print(f"\n--- 联邦训练轮次 {r + 1} ---")
#         local_weights = []
#         total_samples = 0
#
#         for client in clients:
#             if client in client_loaders:
#                 local_model = TrajectoryLSTM(input_size, hidden_size, num_layers, output_size).to(device)
#                 local_model.load_state_dict(global_model.state_dict())
#                 local_weight = train_local(client, local_model, client_loaders[client])
#                 num_samples = len(client_loaders[client].dataset)
#                 total_samples += num_samples
#                 local_weights.append((num_samples, local_weight))
#
#         avg_weights = OrderedDict()
#         for key in global_model.state_dict().keys():
#             weighted_sum = sum(w[key] * n for n, w in local_weights)
#             avg_weights[key] = weighted_sum / total_samples
#
#         global_model.load_state_dict(avg_weights)
#         print(f"联邦轮次 {r + 1} 结束，全局模型已更新")
#         test_model(global_model, val_loader, dataset_type="验证集", record_loss=True)
#
# # ------------------------------
# # 7. 模型测试与评估
# # ------------------------------
# def test_model(model, test_loader, dataset_type="测试集", record_loss=False):
#     model.eval()
#     criterion = nn.MSELoss()
#     test_loss = 0
#     all_preds = []
#     all_targets = []
#
#     with torch.no_grad():
#         for X_batch, Y_batch in test_loader:
#             X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
#             outputs = model(X_batch)
#             loss = criterion(outputs, Y_batch)
#             test_loss += loss.item()
#             all_preds.append(outputs.cpu().numpy())
#             all_targets.append(Y_batch.cpu().numpy())
#
#     test_loss /= len(test_loader)
#     print(f"{dataset_type} MSE损失: {test_loss:.6f}")
#
#     preds = np.concatenate(all_preds, axis=0)
#     targets = np.concatenate(all_targets, axis=0)
#     mae = mean_absolute_error(targets, preds)
#     r2 = r2_score(targets, preds)
#
#     print(f"{dataset_type} MAE: {mae:.6f}")
#     print(f"{dataset_type} R² Score: {r2:.6f}")
#
#     if record_loss:
#         val_losses.append(test_loss)
#
# # ------------------------------
# # 8. 执行训练与测试
# # ------------------------------
# federated_training(global_model, clients, client_loaders, val_loader, rounds=10)
#
# X_test, Y_test = prepare_data(test_data)
# test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)
# test_model(global_model, test_loader, dataset_type="测试集")
#
# # ------------------------------
# # 9. 可视化验证集损失变化
# # ------------------------------
# plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o')
# plt.title("联邦训练中验证集损失变化")
# plt.xlabel("联邦轮次")
# plt.ylabel("验证集 MSE Loss")
# plt.grid(True)
# plt.show()
