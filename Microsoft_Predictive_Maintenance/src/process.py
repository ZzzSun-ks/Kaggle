import pandas as pd
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import TensorDataset, DataLoader

import config

def split_by_machine(df, train_ratio=0.8):
    train_list, test_list = [], []
    for id, data in df.groupby('machineID'):
        data = data.sort_values('datetime')
        split_idx = int(len(data) * train_ratio)
        train_list.append(data.iloc[:split_idx])
        test_list.append(data.iloc[split_idx:])
    return pd.concat(train_list), pd.concat(test_list)

def process():
    print("Process started")
    
    # 加载数据集
    errors = pd.read_csv(config.RAW_DATA_DIR / "PdM_errors.csv")
    failures = pd.read_csv(config.RAW_DATA_DIR / "PdM_failures.csv")
    machines = pd.read_csv(config.RAW_DATA_DIR / "PdM_machines.csv")
    maint = pd.read_csv(config.RAW_DATA_DIR / "PdM_maint.csv")
    telemetry = pd.read_csv(config.RAW_DATA_DIR / "PdM_telemetry.csv")

    # 日期格式
    date_tables = [errors, failures, maint, telemetry]
    for df in date_tables:
        df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S")
        df.sort_values(['machineID', 'datetime'], inplace=True, ignore_index=True)

    # 合并数据集——machines
    dataset = pd.merge(left=machines, right=telemetry,
                        on=['machineID']).sort_values(['datetime', 'machineID'])

    # 合并数据集——errors
    errors.sort_values(['datetime', 'machineID'], inplace=True)
    dataset = pd.merge(left=dataset, right=errors,
                        how='left',
                        on=['datetime', 'machineID'])

    # 合并数据集——maint
    maint.sort_values(['datetime','machineID'], inplace=True)
    dataset = pd.merge(left=dataset, right=maint,
                        how='left',
                        on=['datetime', 'machineID'])

    # 合并数据集——failures（计算RUL）
    failures['RUL'] = failures['datetime']
    failures.sort_values(['datetime', 'machineID'], inplace=True)
    dataset = pd.merge_asof(left=dataset, right=failures[['datetime', 'machineID', 'RUL']],
                            on='datetime',
                            by='machineID',
                            direction='forward').sort_values(['machineID','datetime']).reset_index(drop=True)
    dataset['RUL'] = (dataset['RUL'] - dataset['datetime']).dt.total_seconds() / (24 * 3600)
    dataset['RUL'] = dataset['RUL'].fillna(365)

    # 划分数据集
    train, test = split_by_machine(dataset)

    # 特征预处理
    numerical_features = ['volt', 'rotate', 'pressure', 'vibration']
    categorical_features = ['errorID', 'comp']

    numerical_transformer = Pipeline(
        steps=[
            ("fillna", SimpleImputer(strategy='mean')),
            ("std", StandardScaler())
            ]
        )
    
    categorical_transformer = Pipeline(
        steps=[
            ("fillna", SimpleImputer(strategy="constant", fill_value="NaN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]
        )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
            ]
        )
    
    x_train = pd.DataFrame(preprocessor.fit_transform(train), columns=preprocessor.get_feature_names_out())
    x_test = pd.DataFrame(preprocessor.fit_transform(test), columns=preprocessor.get_feature_names_out())
    y_train = train['RUL']
    y_test = test['RUL']

    train_dataset = TensorDataset(torch.tensor(x_train.values).float(), torch.tensor(y_train.values).float())
    test_dataset = TensorDataset(torch.tensor(x_test.values).float(), torch.tensor(y_test.values).float())

    # 保存数据集
    train_dataset.to_csv(config.PROCESSED_DATA_DIR / 'train.csv', index=False)
    test_dataset.to_csv(config.PROCESSED_DATA_DIR / 'test.csv', index=False)

    print("Process finished")

    return train_dataset, test_dataset, x_train.shape[1]


train_dataset, train_dataset, feature_num = process()

model = nn.Sequential(
    nn.LSTM(feature_num,
            config.HIDDEN_SIZE,
            num_layers=1,
            ))

if __name__ == '__main__':
    process()
