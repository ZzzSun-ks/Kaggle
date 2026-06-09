import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config

# 将预处理后的2D数据重塑为3D数据
def create_rolling_window(data, seq_len=7):
    """
    输入
    data: Dataframe, 包含特征列+machineID+datetime+RUL
    seq_len: int, 序列长度(时间长度)
    输出
    input: np.ndarray shape:[batch, seq_len, feature_num]
        batch: 总的批数
        seq_len: 时间窗口长度
        feature_num: 特征维度
    target: np.ndarray shape:[batch, 1]
        每批数据对应的RUL
    """
    input, target = [], []
    for machine_id in data['__machineID'].unique():
        machine_data = data[data['__machineID'] == machine_id].sort_values(['__datetime'])

        feature_cols = [col for col in machine_data.columns if col not in ['__machineID', '__datetime', 'target__RUL']]
        features = machine_data[feature_cols].values

        rul = machine_data['target__RUL'].values

        for i in range(len(machine_data) - seq_len + 1):
            input.append(features[i:i+seq_len])
            target.append(rul[i+seq_len-1])

    return np.array(input), np.array(target).reshape(-1, 1)

# 1.定义Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, file_path, seq_len=7):
        self.data = pd.read_csv(file_path)
        self.seq_len = seq_len
        self.input, self.target = create_rolling_window(self.data, seq_len=seq_len)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input_tensor = torch.tensor(self.input[index], dtype=torch.float)
        target_tensor = torch.tensor(self.target[index], dtype=torch.float)
        return input_tensor, target_tensor


# 2.提供一个获取dataloader的方法
def get_dataloader(train=True, seq_len=7):
    file_name = 'train.csv' if train else 'test.csv'
    dataset = TimeSeriesDataset(config.PROCESSED_DATA_DIR / file_name, seq_len=seq_len)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(train=False)
    print(len(train_dataloader))
    print(len(test_dataloader))

    for input_tensor, target_tensor in train_dataloader:
        print(input_tensor.shape) # [batch_size, seq_len]
        print(target_tensor.shape) # [batch_size]
        break
