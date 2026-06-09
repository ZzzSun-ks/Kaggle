import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

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


    machines = dataset['machineID'].unique()
    selected_machines = pd.Series(machines).sample(frac=0.05, random_state=42)
    dataset = dataset[dataset['machineID'].isin(selected_machines)]
    print(len(dataset))


    # 划分数据集
    train, test = split_by_machine(dataset)

    # 特征预处理
    train, test = split_by_machine(dataset, train_ratio=0.8)

    # 特征预处理
    nomodify = ['machineID', 'datetime']
    target = ['RUL']
    numerical_features = [col for col in train.select_dtypes(include='float64').columns
                        if col not in nomodify + target]
    categorical_features = [col for col in train.select_dtypes(include='str').columns
                            if col not in nomodify + target]

    preprocessor = ColumnTransformer(transformers=[("", "passthrough", nomodify),
                                                   ("target", SimpleImputer(strategy="constant", fill_value=365), target),
                                                   ("num", StandardScaler(), numerical_features),
                                                   ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
                                                   ],
                                    verbose_feature_names_out=False
                                    )

    preprocessor.fit(train)
    train = pd.DataFrame(preprocessor.transform(train), columns=preprocessor.get_feature_names_out())
    test = pd.DataFrame(preprocessor.transform(test), columns=preprocessor.get_feature_names_out())

    # 保存数据集
    train.to_csv(config.PROCESSED_DATA_DIR / 'train.csv', index=False)
    test.to_csv(config.PROCESSED_DATA_DIR / 'test.csv', index=False)

    print("Process finished")

if __name__ == '__main__':
    process()

