import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os


def process_trajectory_data(file_path, base_path):
    # 设置文件路径
    # base_path = "C:/Users/刘欣逸/Desktop/毕设/"
    # file_path = "C:/Users/刘欣逸/Desktop/毕设/processed_geolife_cleaned.csv"

    # 加载数据
    df = pd.read_csv(file_path)

    # 数据集划分（按 user_id 分割）
    user_ids = df['user_id'].unique()
    train_users, test_users = train_test_split(user_ids, test_size=0.2, random_state=42)
    train_users, val_users = train_test_split(train_users, test_size=0.1, random_state=42)

    train_df = df[df['user_id'].isin(train_users)].copy()
    val_df = df[df['user_id'].isin(val_users)].copy()
    test_df = df[df['user_id'].isin(test_users)].copy()

    # 归一化处理
    scaler = MinMaxScaler()
    features = ['latitude', 'longitude', 'altitude']
    train_df[features] = scaler.fit_transform(train_df[features])
    val_df[features] = scaler.transform(val_df[features])
    test_df[features] = scaler.transform(test_df[features])

    # 保存归一化后的数据
    train_df.to_csv(os.path.join(base_path, "train_data.csv"), index=False)
    val_df.to_csv(os.path.join(base_path, "val_data.csv"), index=False)
    test_df.to_csv(os.path.join(base_path, "test_data.csv"), index=False)

    # 构造时间序列数据，同时保留 user_id 信息
    def create_sequences(data, seq_length=20):
        sequences = []
        user_ids = []

        user_groups = data.groupby("user_id")

        for user_id, group in user_groups:
            group = group.sort_values("datetime")  # 按时间排序
            values = group[features].values

            if len(values) < seq_length + 1:
                continue

            for i in range(len(values) - seq_length):
                seq = values[i:i + seq_length]
                sequences.append(seq)
                user_ids.append(user_id)

        return np.array(sequences), np.array(user_ids)

    # 生成数据
    train_sequences, train_user_ids = create_sequences(train_df)
    val_sequences, val_user_ids = create_sequences(val_df)
    test_sequences, test_user_ids = create_sequences(test_df)

    # 保存序列及 user_id
    np.save(os.path.join(base_path, "train_sequences.npy"), train_sequences)
    np.save(os.path.join(base_path, "train_user_ids.npy"), train_user_ids)

    np.save(os.path.join(base_path, "val_sequences.npy"), val_sequences)
    np.save(os.path.join(base_path, "val_user_ids.npy"), val_user_ids)

    np.save(os.path.join(base_path, "test_sequences.npy"), test_sequences)
    np.save(os.path.join(base_path, "test_user_ids.npy"), test_user_ids)

    print("✅ 数据预处理完成，序列与用户信息已保存！")

    return True, "✅ 数据预处理完成，序列与用户信息已保存！"

    # except Exception as e:
    # return False, f"❌ 处理失败: {str(e)}"
