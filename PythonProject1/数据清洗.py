import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from io import BytesIO
import base64
import os

def data_clean(input_path, output_path):
    """
    数据清洗与可视化函数
    参数:
        input_path: C:/Users/刘欣逸/Desktop/毕设/processed_data/processed_geolife_all.pkl
        output_path: C:/Users/刘欣逸/Desktop/毕设/
    返回:
        {
            "original_count": 原始数据量,
            "cleaned_count": 清洗后数据量,
            "downsampled_count": 降采样后数据量,
            "geo_plot": 地理分布图base64,
            "time_plot": 时间分布图base64
        }
    """
    # === 初始化设置 ===
    rcParams['font.family'] = 'SimHei'
    rcParams['axes.unicode_minus'] = False

    # === 读取数据 ===
    df = pd.read_pickle(input_path)
    original_count = len(df)

    # === 生成可视化图表 ===
    plots = {}

    # 地理分布图
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df["longitude"], y=df["latitude"], hue=df["user_id"],
                   palette="tab10", alpha=0.5, s=5)
    plt.title("轨迹点地理分布")
    plt.xlabel("经度")
    plt.ylabel("纬度")
    plt.legend([], [], frameon=False)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plots["geo_plot"] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # 时间分布图
    df["hour"] = df["datetime"].dt.hour
    plt.figure(figsize=(8, 4))
    sns.histplot(df["hour"], bins=24, kde=True)
    plt.title("轨迹点时间分布")
    plt.xlabel("小时")
    plt.ylabel("轨迹点数量")
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plots["time_plot"] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # === 数据清洗 ===
    df_cleaned = df.dropna(subset=["latitude", "longitude", "altitude"])

    # 计算移动参数
    df_cleaned["time_diff"] = df_cleaned.groupby("user_id")["datetime"].diff().dt.total_seconds()
    df_cleaned["lat_diff"] = df_cleaned.groupby("user_id")["latitude"].diff()
    df_cleaned["lon_diff"] = df_cleaned.groupby("user_id")["longitude"].diff()

    # 计算距离和速度
    R = 6371000  # 地球半径（米）
    df_cleaned["distance"] = np.sqrt(
        (df_cleaned["lat_diff"] * R * np.pi / 180) ** 2 +
        (df_cleaned["lon_diff"] * R * np.pi / 180) ** 2
    )
    df_cleaned["speed"] = df_cleaned["distance"] / df_cleaned["time_diff"]

    # 过滤异常速度
    df_cleaned = df_cleaned[(df_cleaned["speed"] < 50) | df_cleaned["speed"].isna()]

    # 删除临时计算列
    df_cleaned.drop(columns=["time_diff", "lat_diff", "lon_diff", "distance", "speed"], inplace=True)

    cleaned_count = len(df_cleaned)

    # === 降采样 ===
    df_downsampled = df_cleaned.groupby("user_id").apply(lambda x: x.iloc[::5])
    downsampled_count = len(df_downsampled)

    # === 保存结果 ===
    df_downsampled.to_csv(os.path.join(output_path, "processed_geolife_cleaned.csv"), index=False, encoding='utf-8')
    df_downsampled.to_pickle(os.path.join(output_path, "processed_geolife_cleaned.pkl"))

    return {
        "original_count": original_count,
        "cleaned_count": cleaned_count,
        "downsampled_count": downsampled_count,
        "geo_plot": plots["geo_plot"],
        "time_plot": plots["time_plot"]
    }

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib import rcParams
#
# # 数据清洗函数
# def dataclean():
#     # === 1. 设置字体，确保中文显示 ===
#     rcParams['font.family'] = 'SimHei'  # 使用黑体
#     rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#
#     # === 2. 读取数据 ===
#     file_path = r"C:\Users\刘欣逸\Desktop\毕设\processed_data\processed_geolife_all.pkl"
#     df = pd.read_pickle(file_path)
#
#     # === 3. 轨迹数据可视化 ===
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(x=df["longitude"], y=df["latitude"], hue=df["user_id"], palette="tab10", alpha=0.5, s=5)
#     plt.title("轨迹点地理分布")
#     plt.xlabel("经度")
#     plt.ylabel("纬度")
#     plt.legend([],[], frameon=False)  # 隐藏混乱图例
#     plt.show()
#
#     df["hour"] = df["datetime"].dt.hour
#     plt.figure(figsize=(8, 4))
#     sns.histplot(df["hour"], bins=24, kde=True)
#     plt.title("轨迹点时间分布")
#     plt.xlabel("小时")
#     plt.ylabel("轨迹点数量")
#     plt.show()
#
#     # === 4. 数据清洗 ===
#     df.dropna(subset=["latitude", "longitude", "altitude"], inplace=True)  # 删除缺失值
#
#     # 计算时间差
#     df["time_diff"] = df.groupby("user_id")["datetime"].diff().dt.total_seconds()
#     df["lat_diff"] = df.groupby("user_id")["latitude"].diff()
#     df["lon_diff"] = df.groupby("user_id")["longitude"].diff()
#
#     # 计算近似地理距离
#     R = 6371000  # 地球半径（米）
#     df["distance"] = np.sqrt((df["lat_diff"] * R * np.pi / 180) ** 2 + (df["lon_diff"] * R * np.pi / 180) ** 2)
#
#     # 计算速度（m/s）
#     df["speed"] = df["distance"] / df["time_diff"]
#
#     # 过滤异常值：移除速度 > 50m/s (180km/h) 的数据
#     df_cleaned = df[(df["speed"] < 50) | df["speed"].isna()].copy()
#
#     # 删除临时计算列
#     df_cleaned.drop(columns=["time_diff", "lat_diff", "lon_diff", "distance", "speed"], inplace=True)
#
#     print(f"清洗后数据量：{len(df_cleaned)}")
#
#     # === 5. 降采样（每隔5s取一个点） ===
#     df_downsampled = df_cleaned.groupby("user_id").apply(lambda x: x.iloc[::5]).reset_index(drop=True)
#
#     # 保存数据
#     output_path = "C:/Users/刘欣逸/Desktop/毕设/processed_geolife_cleaned.pkl"
#     output_path1 = "C:/Users/刘欣逸/Desktop/毕设/processed_geolife_cleaned.csv"
#     df_downsampled.to_pickle(output_path)
#     df_downsampled.to_csv(output_path1, index=False, encoding='utf-8')  # 保存为CSV，并设置编码为utf-8
#     print(f"降采样后数据量：{len(df_downsampled)}")
#     print(f"数据已保存至 {output_path}")
#
# dataclean()