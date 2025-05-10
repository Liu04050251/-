
import os
import pandas as pd
from tqdm import tqdm

def datapreprocessing(data_root, save_dir, progress_callback):
    """
    处理轨迹数据的函数，处理过程中会调用回调函数报告进度。

    参数:
    - data_root: 轨迹数据根目录
    - save_dir: 保存处理后数据的目录
    - progress_callback: 进度回调函数

    返回:
    - 处理后的 DataFrame
    """
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

    # 获取所有 .plt 文件路径
    trajectory_files = []
    for user_folder in os.listdir(data_root):
        traj_path = os.path.join(data_root, user_folder, "Trajectory")
        if os.path.exists(traj_path):
            trajectory_files.extend([os.path.join(traj_path, f) for f in os.listdir(traj_path) if f.endswith(".plt")])

    # 初始化一个空的 DataFrame 来存储所有数据
    all_data = []

    # 逐个文件处理
    for i, file_path in enumerate(tqdm(trajectory_files, desc="Processing Trajectories")):
        # 读取数据（跳过前6行元数据）
        df = pd.read_csv(file_path, skiprows=6, header=None)

        # 设置列名
        df.columns = ["latitude", "longitude", "zero", "altitude", "time", "date", "clock_time"]

        # 删除无用列
        df.drop(columns=["zero"], inplace=True)

        # 合并 date 和 time
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["clock_time"], format="%Y-%m-%d %H:%M:%S").dt.floor("s")

        # 删除原始时间列
        df.drop(columns=["date", "time", "clock_time"], inplace=True)

        # 按时间排序
        df = df.sort_values(by="datetime").reset_index(drop=True)

        # 添加一个列用于标识用户（从文件路径中提取用户ID）
        user_id = os.path.basename(os.path.dirname(os.path.dirname(file_path)))  # 获取用户 ID
        df["user_id"] = user_id

        # 添加到总数据列表
        all_data.append(df)

        # 调用进度回调，报告进度
        progress_callback(i + 1, len(trajectory_files))

    # 合并所有数据
    final_df = pd.concat(all_data, ignore_index=True)

    # 保存处理后的数据
    processed_file_path = os.path.join(save_dir, "processed_geolife_all.csv")
    final_df.to_csv(processed_file_path, index=False)
    final_df.to_pickle(os.path.join(save_dir, "processed_geolife_all.pkl"))

    print(f"数据处理完成，共处理 {len(trajectory_files)} 个文件，最终数据包含 {len(final_df)} 条轨迹点。")

    # 返回处理后的文件路径和数据
    return processed_file_path, final_df


#
# import pandas as pd
# import os
# from tqdm import tqdm  # 用于进度条显示
#
# # 设置轨迹数据的根目录
# data_root = "C:/Users/刘欣逸/Desktop/毕设/Geolife Trajectories 1.3/Data/"
#
# # 设置保存处理后数据的目录
# save_dir = "C:/Users/刘欣逸/Desktop/毕设/processed_data/"
# os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
#
# # 获取所有 .plt 文件路径
# trajectory_files = []
# for user_folder in os.listdir(data_root):
#     traj_path = os.path.join(data_root, user_folder, "Trajectory")
#     if os.path.exists(traj_path):
#         trajectory_files.extend([os.path.join(traj_path, f) for f in os.listdir(traj_path) if f.endswith(".plt")])
#
# # 初始化一个空的 DataFrame 来存储所有数据
# all_data = []
#
# # 逐个文件处理
# for file_path in tqdm(trajectory_files, desc="Processing Trajectories"):
#     # 读取数据（跳过前6行元数据）
#     df = pd.read_csv(file_path, skiprows=6, header=None)
#
#     # 设置列名
#     df.columns = ["latitude", "longitude", "zero", "altitude", "time", "date", "clock_time"]
#
#     # 删除无用列
#     df.drop(columns=["zero"], inplace=True)
#
#     # 合并 date 和 time
#     df["datetime"] = pd.to_datetime(df["date"] + " " + df["clock_time"],format="%Y-%m-%d %H:%M:%S").dt.floor("s")
#
#     # 删除原始时间列
#     df.drop(columns=["date", "time", "clock_time"], inplace=True)
#
#     # 按时间排序
#     df = df.sort_values(by="datetime").reset_index(drop=True)
#
#     # 添加一个列用于标识用户（从文件路径中提取用户ID）
#     user_id = os.path.basename(os.path.dirname(os.path.dirname(file_path)))  # 获取用户 ID
#     df["user_id"] = user_id
#
#     # 添加到总数据列表
#     all_data.append(df)
#
# # 合并所有数据
# final_df = pd.concat(all_data, ignore_index=True)
#
# # 保存处理后的数据
# final_df.to_csv(os.path.join(save_dir, "processed_geolife_all.csv"), index=False)
# final_df.to_pickle(os.path.join(save_dir, "processed_geolife_all.pkl"))
#
# print(f"数据处理完成，共处理 {len(trajectory_files)} 个文件，最终数据包含 {len(final_df)} 条轨迹点。")
