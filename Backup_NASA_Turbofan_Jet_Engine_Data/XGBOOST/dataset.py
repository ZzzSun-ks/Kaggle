import pandas as pd
import config

def load_data(filename):
    # 创建列名
    index_names = ['unit_number', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i+1) for i in range(0, 21)]
    col_names = index_names + setting_names + sensor_names
    
    # 加载数据集
    df = pd.read_csv(config.RAW_DATA_DIR / filename,
                     sep=r'\s+', 
                     header=None, 
                     index_col=False, 
                     names=col_names)

    return df

if __name__ == "__main__":
    df = load_data('train_FD003.txt')
    print(df.head())