import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, mean_absolute_error
from io import BytesIO
import base64
import os

def visualize_results(model_dir):

    try:
        # 1. 设备设置
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2. 加载数据
        #base_path = "C:/Users/刘欣逸/Desktop/毕设/"
        base_path = os.path.dirname(model_dir)
        test_file = os.path.join(base_path, "test_sequences.npy")
        test_data = np.load(test_file)

        # 3. 数据预处理
        def prepare_data(data):
            X = data[:, :-1, :]  # 取前 (seq_length-1) 作为输入
            Y = data[:, -1, :]  # 取最后一个时间步作为预测目标
            return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

        X_test, Y_test = prepare_data(test_data)

        # 4. 定义 LSTM 模型
        class TrajectoryLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(TrajectoryLSTM, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
                return out

        # 5. 初始化模型
        input_size = X_test.shape[2]
        hidden_size = 32
        num_layers = 1
        output_size = input_size

        model = TrajectoryLSTM(input_size, hidden_size, num_layers, output_size).to(device)

        # 加载模型
        model.load_state_dict(torch.load(model_dir, map_location=device))
        model.eval()

        # 释放 GPU 缓存
        torch.cuda.empty_cache()

        # 6. 进行预测
        predictions = []
        with torch.no_grad():
            for i in range(X_test.shape[0]):
                x_input = X_test[i].unsqueeze(0).to(device, non_blocking=True)
                y_pred = model(x_input).cpu().numpy()
                predictions.append(y_pred)

        predictions = np.array(predictions).squeeze()

        # 7. 计算误差指标
        mse = mean_squared_error(Y_test, predictions)
        mae = mean_absolute_error(Y_test, predictions)
        rmse = np.sqrt(mse)

        print(f"均方误差 (MSE): {mse:.6f}")
        print(f"平均绝对误差 (MAE): {mae:.6f}")
        print(f"均方根误差 (RMSE): {rmse:.6f}")

        # 8. 可视化
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 真实轨迹
        ax.plot(Y_test[:, 0], Y_test[:, 1], Y_test[:, 2], label='True Trajectory', color='b')

        # 预测轨迹
        ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], label='Predicted Trajectory', color='r')

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Altitude')
        ax.legend()
        #plt.show()

        # 转换为base64
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "plot": plot_base64
        }

    except Exception as e:
        return {"error": f"处理失败: {str(e)}"}


# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.metrics import mean_squared_error, mean_absolute_error
#
# # 1. 设备设置
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 2. 加载数据
# base_path = "C:/Users/刘欣逸/Desktop/毕设/"
# test_file = base_path + "test_sequences.npy"
# test_data = np.load(test_file)
#
# # 3. 数据预处理
# def prepare_data(data):
#     X = data[:, :-1, :]  # 取前 (seq_length-1) 作为输入
#     Y = data[:, -1, :]  # 取最后一个时间步作为预测目标
#     return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
#
# X_test, Y_test = prepare_data(test_data)
#
# # 4. 定义 LSTM 模型
# class TrajectoryLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(TrajectoryLSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
#         return out
#
# # 5. 初始化模型
# input_size = X_test.shape[2]
# hidden_size = 32
# num_layers = 1
# output_size = input_size
#
# model = TrajectoryLSTM(input_size, hidden_size, num_layers, output_size).to(device)
#
# # 加载模型
# model.load_state_dict(torch.load(base_path + "best_lstm_model.pth", map_location=device))
# model.eval()
#
# # 释放 GPU 缓存
# torch.cuda.empty_cache()
#
# # 6. 进行预测
# predictions = []
# with torch.no_grad():
#     for i in range(X_test.shape[0]):
#         x_input = X_test[i].unsqueeze(0).to(device, non_blocking=True)
#         y_pred = model(x_input).cpu().numpy()
#         predictions.append(y_pred)
#
# predictions = np.array(predictions).squeeze()
#
# # 7. 计算误差指标
# mse = mean_squared_error(Y_test, predictions)
# mae = mean_absolute_error(Y_test, predictions)
# rmse = np.sqrt(mse)
#
# print(f"均方误差 (MSE): {mse:.6f}")
# print(f"平均绝对误差 (MAE): {mae:.6f}")
# print(f"均方根误差 (RMSE): {rmse:.6f}")
#
# # 8. 可视化
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 真实轨迹
# ax.plot(Y_test[:, 0], Y_test[:, 1], Y_test[:, 2], label='True Trajectory', color='b')
#
# # 预测轨迹
# ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], label='Predicted Trajectory', color='r')
#
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# ax.set_zlabel('Altitude')
# ax.legend()
# plt.show()
#


