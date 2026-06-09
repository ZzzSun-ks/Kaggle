import pandas as pd
import numpy as np
from typing import List, Union, Optional
from datetime import datetime, timedelta

def generate_features(
    data: pd.DataFrame,
    machine_col: str,
    time_col: str,
    target_col: str,
    lag_periods: Union[int, List[int]] = 3,
    rolling_windows: Union[int, List[int]] = 5,
    include_rolling_stats: bool = True,
    rolling_stats: List[str] = ['mean', 'std', 'min', 'max'],
    fill_na_method: str = 'drop',
    fill_value: Optional[float] = None,
    fill_by_group: bool = True
) -> pd.DataFrame:
    """
    自动生成时间序列特征数据集
    
    Parameters:
    -----------
    data : pd.DataFrame
        原始数据集，必须包含机器编号列、时间列和目标列
    machine_col : str
        机器编号列名
    time_col : str
        时间列名
    target_col : str
        目标变量列名
    lag_periods : int or List[int], default=3
        滞后阶数，可以是整数(生成1到该整数的所有滞后)或具体的滞后阶数列表
    rolling_windows : int or List[int], default=5
        滚动窗口长度，可以是整数或列表
    include_rolling_stats : bool, default=True
        是否包含滚动统计特征
    rolling_stats : List[str], default=['mean', 'std', 'min', 'max']
        需要计算的滚动统计量，可选：'mean', 'std', 'min', 'max', 'median', 'sum', 'var', 'skew', 'kurt'
    fill_na_method : str, default='drop'
        缺失值处理方法：
        - 'drop': 删除含有缺失值的行
        - 'ffill': 前向填充
        - 'bfill': 后向填充
        - 'zero': 填充0
        - 'mean': 填充均值
        - 'median': 填充中位数
        - 'value': 填充指定值
        - 'interpolate': 线性插值
        - 'ffill_bfill': 先向前填充，剩余缺失值再向后填充
        - 'bfill_ffill': 先向后填充，剩余缺失值再向前填充
    fill_value : float, optional
        当fill_na_method='value'时使用的填充值
    fill_by_group : bool, default=True
        是否按机器分组计算均值/中位数，False则使用全局均值/中位数
    
    Returns:
    --------
    pd.DataFrame : 包含原始列和生成特征的数据集
    """
    
    # 参数处理
    if isinstance(lag_periods, int):
        lag_periods = list(range(1, lag_periods + 1))
    
    if isinstance(rolling_windows, int):
        rolling_windows = [rolling_windows]
    
    # 数据验证
    required_cols = [machine_col, time_col, target_col]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"列 '{col}' 不在数据集中")
    
    # 创建数据副本
    df = data.copy()
    
    # 确保时间列是datetime类型
    df[time_col] = pd.to_datetime(df[time_col])
    
    # 按机器编号和时间排序
    df = df.sort_values([machine_col, time_col]).reset_index(drop=True)
    
    # 存储所有特征列名
    feature_cols = []
    
    print("=" * 60)
    print("开始生成特征...")
    print(f"机器数量: {df[machine_col].nunique()}")
    print(f"数据行数: {len(df)}")
    print(f"滞后阶数: {lag_periods}")
    print(f"滚动窗口: {rolling_windows}")
    print(f"缺失值处理方法: {fill_na_method}")
    print("=" * 60)
    
    # 1. 生成滞后特征
    print("\n1. 生成滞后特征...")
    for lag in lag_periods:
        col_name = f'{target_col}_lag_{lag}'
        df[col_name] = df.groupby(machine_col)[target_col].shift(lag)
        feature_cols.append(col_name)
        print(f"   - 已生成: {col_name}")
    
    # 2. 生成差分特征
    print("\n2. 生成差分特征...")
    for diff_period in [1, 2]:
        col_name = f'{target_col}_diff_{diff_period}'
        df[col_name] = df.groupby(machine_col)[target_col].diff(diff_period)
        feature_cols.append(col_name)
        print(f"   - 已生成: {col_name}")
    
    # 3. 生成滚动窗口特征
    if include_rolling_stats and rolling_windows:
        print("\n3. 生成滚动窗口特征...")
        
        for window in rolling_windows:
            for stat in rolling_stats:
                col_name = f'{target_col}_rolling_{window}_{stat}'
                
                try:
                    if stat == 'kurt':
                        df[col_name] = df.groupby(machine_col)[target_col].transform(
                            lambda x: x.rolling(window=window, min_periods=1).kurt()
                        )
                    elif stat == 'skew':
                        df[col_name] = df.groupby(machine_col)[target_col].transform(
                            lambda x: x.rolling(window=window, min_periods=1).skew()
                        )
                    else:
                        df[col_name] = df.groupby(machine_col)[target_col].transform(
                            lambda x: x.rolling(window=window, min_periods=1).agg(stat)
                        )
                    
                    feature_cols.append(col_name)
                    print(f"   - 已生成: {col_name}")
                    
                except Exception as e:
                    print(f"   - 警告: 生成 {col_name} 失败: {e}")
    
    # 4. 生成时间特征
    print("\n4. 生成时间特征...")
    if time_col in df.columns:
        time_features = {
            'hour': df[time_col].dt.hour,
            'dayofweek': df[time_col].dt.dayofweek,
            'month': df[time_col].dt.month,
            'quarter': df[time_col].dt.quarter,
            'dayofyear': df[time_col].dt.dayofyear,
        }
        
        for feat_name, feat_values in time_features.items():
            col_name = f'{time_col}_{feat_name}'
            df[col_name] = feat_values
            feature_cols.append(col_name)
        
        print(f"   - 已生成时间特征: {', '.join(time_features.keys())}")
    
    # 5. 生成机器级别的统计特征
    print("\n5. 生成机器级别统计特征...")
    machine_stats = df.groupby(machine_col)[target_col].agg(['mean', 'std', 'min', 'max', 'median']).reset_index()
    machine_stats.columns = [machine_col, 
                            f'{target_col}_machine_mean', 
                            f'{target_col}_machine_std', 
                            f'{target_col}_machine_min', 
                            f'{target_col}_machine_max',
                            f'{target_col}_machine_median']
    df = df.merge(machine_stats, on=machine_col, how='left')
    feature_cols.extend([col for col in machine_stats.columns if col != machine_col])
    print(f"   - 已生成机器级别特征: mean, std, min, max, median")
    
    # 6. 处理缺失值
    print(f"\n6. 处理缺失值...")
    print(f"   处理方法: {fill_na_method}")
    
    original_len = len(df)
    original_nan_count = df[feature_cols].isna().sum().sum()
    print(f"   缺失值总数: {original_nan_count}")
    
    if fill_na_method == 'drop':
        # 删除含有缺失值的行
        df = df.dropna(subset=feature_cols)
        print(f"   - 删除了 {original_len - len(df)} 行含有缺失值的数据")
        
    elif fill_na_method == 'ffill':
        # 前向填充
        df[feature_cols] = df.groupby(machine_col)[feature_cols].transform(lambda x: x.ffill())
        remaining_nan = df[feature_cols].isna().sum().sum()
        if remaining_nan > 0:
            print(f"   - 前向填充后仍有 {remaining_nan} 个缺失值，用0填充")
            df[feature_cols] = df[feature_cols].fillna(0)
            
    elif fill_na_method == 'bfill':
        # 后向填充
        df[feature_cols] = df.groupby(machine_col)[feature_cols].transform(lambda x: x.bfill())
        remaining_nan = df[feature_cols].isna().sum().sum()
        if remaining_nan > 0:
            print(f"   - 后向填充后仍有 {remaining_nan} 个缺失值，用0填充")
            df[feature_cols] = df[feature_cols].fillna(0)
            
    elif fill_na_method == 'zero':
        # 填充0
        df[feature_cols] = df[feature_cols].fillna(0)
        print(f"   - 用0填充了 {original_nan_count} 个缺失值")
        
    elif fill_na_method == 'mean':
        # 均值填充
        if fill_by_group:
            # 按机器分组计算均值
            print(f"   - 使用按机器分组的均值填充")
            group_means = df.groupby(machine_col)[feature_cols].transform('mean')
            df[feature_cols] = df[feature_cols].fillna(group_means)
        else:
            # 使用全局均值
            print(f"   - 使用全局均值填充")
            global_means = df[feature_cols].mean()
            df[feature_cols] = df[feature_cols].fillna(global_means)
        
        # 处理仍存在的缺失值（如果某列全为NaN）
        remaining_nan = df[feature_cols].isna().sum().sum()
        if remaining_nan > 0:
            print(f"   - 均值填充后仍有 {remaining_nan} 个缺失值，用0填充")
            df[feature_cols] = df[feature_cols].fillna(0)
            
    elif fill_na_method == 'median':
        # 中位数填充
        if fill_by_group:
            # 按机器分组计算中位数
            print(f"   - 使用按机器分组的中位数填充")
            group_medians = df.groupby(machine_col)[feature_cols].transform('median')
            df[feature_cols] = df[feature_cols].fillna(group_medians)
        else:
            # 使用全局中位数
            print(f"   - 使用全局中位数填充")
            global_medians = df[feature_cols].median()
            df[feature_cols] = df[feature_cols].fillna(global_medians)
        
        # 处理仍存在的缺失值
        remaining_nan = df[feature_cols].isna().sum().sum()
        if remaining_nan > 0:
            print(f"   - 中位数填充后仍有 {remaining_nan} 个缺失值，用0填充")
            df[feature_cols] = df[feature_cols].fillna(0)
            
    elif fill_na_method == 'value':
        # 用指定值填充
        if fill_value is None:
            fill_value = 0
            print(f"   - 未指定填充值，默认使用0填充")
        print(f"   - 使用固定值 {fill_value} 填充")
        df[feature_cols] = df[feature_cols].fillna(fill_value)
        
    elif fill_na_method == 'interpolate':
        # 线性插值
        print(f"   - 使用线性插值填充")
        df[feature_cols] = df.groupby(machine_col)[feature_cols].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )
        # 处理插值后仍存在的缺失值（首尾可能无法插值）
        remaining_nan = df[feature_cols].isna().sum().sum()
        if remaining_nan > 0:
            print(f"   - 插值后仍有 {remaining_nan} 个缺失值，用前向/后向填充")
            df[feature_cols] = df.groupby(machine_col)[feature_cols].transform(lambda x: x.ffill().bfill())
            
    elif fill_na_method == 'ffill_bfill':
        # 先向前填充，剩余缺失值再向后填充
        print(f"   - 先向前填充，剩余缺失值再向后填充")
        df[feature_cols] = df.groupby(machine_col)[feature_cols].transform(lambda x: x.ffill())
        remaining_nan = df[feature_cols].isna().sum().sum()
        if remaining_nan > 0:
            print(f"   - 向前填充后仍有 {remaining_nan} 个缺失值，继续向后填充")
            df[feature_cols] = df.groupby(machine_col)[feature_cols].transform(lambda x: x.bfill())
            remaining_nan = df[feature_cols].isna().sum().sum()
            if remaining_nan > 0:
                print(f"   - 仍剩余 {remaining_nan} 个缺失值，用0填充")
                df[feature_cols] = df[feature_cols].fillna(0)
                
    elif fill_na_method == 'bfill_ffill':
        # 先向后填充，剩余缺失值再向前填充
        print(f"   - 先向后填充，剩余缺失值再向前填充")
        df[feature_cols] = df.groupby(machine_col)[feature_cols].transform(lambda x: x.bfill())
        remaining_nan = df[feature_cols].isna().sum().sum()
        if remaining_nan > 0:
            print(f"   - 向后填充后仍有 {remaining_nan} 个缺失值，继续向前填充")
            df[feature_cols] = df.groupby(machine_col)[feature_cols].transform(lambda x: x.ffill())
            remaining_nan = df[feature_cols].isna().sum().sum()
            if remaining_nan > 0:
                print(f"   - 仍剩余 {remaining_nan} 个缺失值，用0填充")
                df[feature_cols] = df[feature_cols].fillna(0)
    
    else:
        raise ValueError(f"不支持的填充方法: {fill_na_method}")
    
    # 最终检查是否还有缺失值
    final_nan_count = df[feature_cols].isna().sum().sum()
    if final_nan_count > 0:
        print(f"   - 警告: 处理后仍有 {final_nan_count} 个缺失值，强制用0填充")
        df[feature_cols] = df[feature_cols].fillna(0)
    
    print(f"   - 最终数据行数: {len(df)} (原始: {original_len})")
    
    # 输出汇总信息
    print("\n" + "=" * 60)
    print("特征生成完成！")
    print(f"生成的特征总数: {len(feature_cols)}")
    print(f"最终数据集形状: {df.shape}")
    print(f"缺失值总数: {df[feature_cols].isna().sum().sum()}")
    print("=" * 60)
    
    return df


def demo_usage():
    """
    演示各种缺失值处理方法
    """
    # 创建示例数据，包含一些缺失值
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    demo_data = []
    for machine_id in [1, 2, 3]:
        for date in dates:
            base_value = machine_id * 10
            noise = np.random.normal(0, 5)
            value = base_value + np.sin(date.day / 10) * 5 + noise
            
            # 随机添加一些缺失值
            if np.random.random() < 0.1:  # 10%的概率为NaN
                value = np.nan
                
            demo_data.append({
                'machine_id': machine_id,
                'date': date,
                'temperature': value
            })
    
    df = pd.DataFrame(demo_data)
    
    print("原始数据预览:")
    print(df.head(10))
    print(f"\n数据集形状: {df.shape}")
    print(f"目标变量缺失值数量: {df['temperature'].isna().sum()}")
    
    # 演示不同的填充方法
    fill_methods = [
        'drop',
        'ffill', 
        'bfill', 
        'zero', 
        'mean', 
        'median', 
        'value', 
        'interpolate', 
        'ffill_bfill',
        'bfill_ffill'
    ]
    
    results = {}
    
    for method in fill_methods:
        print(f"\n{'='*80}")
        print(f"测试填充方法: {method}")
        print('='*80)
        
        try:
            featured_df = generate_features(
                data=df,
                machine_col='machine_id',
                time_col='date',
                target_col='temperature',
                lag_periods=[1, 3, 5],
                rolling_windows=[5],
                include_rolling_stats=True,
                rolling_stats=['mean', 'std'],
                fill_na_method=method,
                fill_value=-999 if method == 'value' else None,
                fill_by_group=True
            )

            print(featured_df.head())
            
            results[method] = {
                'shape': featured_df.shape,
                'nan_count': featured_df.isna().sum().sum()
            }
            print(f"\n✅ {method} 方法成功完成")
            
        except Exception as e:
            print(f"\n❌ {method} 方法失败: {e}")
    
    # 对比不同方法的结果
    print(f"\n{'='*80}")
    print("各填充方法结果对比:")
    print('='*80)
    print(f"{'方法':<15} {'数据形状':<15} {'缺失值数量':<10}")
    print("-" * 40)
    for method, result in results.items():
        print(f"{method:<15} {str(result['shape']):<15} {result['nan_count']:<10}")
    
    return df, results


if __name__ == "__main__":
    try:
        df, results = demo_usage()
        print("\n✅ 所有测试通过！")
        print(f"\n可用填充方法: drop, ffill, bfill, zero, mean, median, value, interpolate, ffill_bfill, bfill_ffill")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
