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
    # 目标变量的缺失值处理
    target_fill_na_method: str = 'interpolate',
    target_fill_by_group: bool = True,
    target_fill_value: Optional[float] = None,
    # 生成特征后的缺失值处理（通常不需要，但保留作为安全措施）
    feature_fill_na_method: str = 'ffill',
    feature_fill_by_group: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    自动生成时间序列特征数据集（优先在特征生成前处理缺失值）
    
    Parameters:
    -----------
    data : pd.DataFrame
        原始数据集
    machine_col : str
        机器编号列名
    time_col : str
        时间列名
    target_col : str
        目标变量列名
    lag_periods : int or List[int], default=3
        滞后阶数
    rolling_windows : int or List[int], default=5
        滚动窗口长度
    include_rolling_stats : bool, default=True
        是否包含滚动统计特征
    rolling_stats : List[str], default=['mean', 'std', 'min', 'max']
        滚动统计量
    target_fill_na_method : str, default='interpolate'
        目标变量缺失值处理方法（在生成特征前）
        可选：'drop', 'ffill', 'bfill', 'zero', 'mean', 'median', 
              'value', 'interpolate', 'ffill_bfill', 'bfill_ffill'
    target_fill_by_group : bool, default=True
        是否按机器分组处理目标变量缺失值
    target_fill_value : float, optional
        当target_fill_na_method='value'时的填充值
    feature_fill_na_method : str, default='ffill'
        特征列缺失值处理方法（在生成特征后，作为安全措施）
    feature_fill_by_group : bool, default=True
        是否按机器分组处理特征列缺失值
    verbose : bool, default=True
        是否输出详细信息
    
    Returns:
    --------
    pd.DataFrame : 包含原始列和生成特征的数据集
    dict : 包含处理统计信息的字典
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
    
    # 统计信息
    stats = {
        'original_rows': len(data),
        'target_nan_before': data[target_col].isna().sum(),
        'target_nan_after': 0,
        'feature_nan_after': 0,
        'final_rows': 0
    }
    
    # 创建数据副本
    df = data.copy()
    
    # 确保时间列是datetime类型
    df[time_col] = pd.to_datetime(df[time_col])
    
    # 按机器编号和时间排序
    df = df.sort_values([machine_col, time_col]).reset_index(drop=True)
    
    if verbose:
        print("=" * 70)
        print("开始生成特征（优先在特征生成前处理目标变量缺失值）")
        print("=" * 70)
        print(f"原始数据: {stats['original_rows']} 行, "
              f"{df[machine_col].nunique()} 台机器")
        print(f"目标变量 '{target_col}' 缺失值: {stats['target_nan_before']}")
        print(f"目标变量缺失值处理方法: {target_fill_na_method}")
        print(f"按机器分组处理: {'是' if target_fill_by_group else '否'}")
    
    # ==========================================
    # 步骤1: 处理目标变量的缺失值（关键步骤！）
    # ==========================================
    if stats['target_nan_before'] > 0:
        print(f"\n{'='*70}")
        print(f"步骤1: 处理目标变量 '{target_col}' 的缺失值")
        print(f"{'='*70}")
        
        df = _fill_missing_values(
            df=df,
            columns=[target_col],
            method=target_fill_na_method,
            group_col=machine_col if target_fill_by_group else None,
            fill_value=target_fill_value,
            verbose=verbose
        )
        
        stats['target_nan_after'] = df[target_col].isna().sum()
        
        if stats['target_nan_after'] > 0:
            if verbose:
                print(f"⚠️  警告: 处理后仍有 {stats['target_nan_after']} 个缺失值")
            # 如果整个组都是NaN，考虑删除
            df = df.dropna(subset=[target_col])
            stats['final_rows'] = len(df)
            if verbose:
                print(f"   删除了 {stats['original_rows'] - stats['final_rows']} 行全NaN数据")
    
    # ==========================================
    # 步骤2: 生成特征
    # ==========================================
    if verbose:
        print(f"\n{'='*70}")
        print("步骤2: 生成特征")
        print(f"{'='*70}")
    
    feature_cols = []
    
    # 2.1 生成滞后特征
    if verbose:
        print(f"\n2.1 滞后特征 (阶数: {lag_periods})")
    for lag in lag_periods:
        col_name = f'{target_col}_lag_{lag}'
        df[col_name] = df.groupby(machine_col)[target_col].shift(lag)
        feature_cols.append(col_name)
        if verbose:
            nan_count = df[col_name].isna().sum()
            print(f"   ✓ {col_name}: {nan_count} 个缺失值")
    
    # 2.2 生成差分特征
    if verbose:
        print(f"\n2.2 差分特征")
    for diff_period in [1, 2]:
        col_name = f'{target_col}_diff_{diff_period}'
        df[col_name] = df.groupby(machine_col)[target_col].diff(diff_period)
        feature_cols.append(col_name)
        if verbose:
            nan_count = df[col_name].isna().sum()
            print(f"   ✓ {col_name}: {nan_count} 个缺失值")
    
    # 2.3 生成滚动窗口特征
    if include_rolling_stats and rolling_windows:
        if verbose:
            print(f"\n2.3 滚动窗口特征 (窗口: {rolling_windows}, 统计量: {rolling_stats})")
        
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
                    if verbose:
                        nan_count = df[col_name].isna().sum()
                        print(f"   ✓ {col_name}: {nan_count} 个缺失值")
                        
                except Exception as e:
                    if verbose:
                        print(f"   ✗ {col_name}: 生成失败 - {e}")
    
    # 2.4 生成时间特征
    if verbose:
        print(f"\n2.4 时间特征")
    
    time_features = {}
    try:
        time_features = {
            'hour': df[time_col].dt.hour,
            'dayofweek': df[time_col].dt.dayofweek,
            'month': df[time_col].dt.month,
            'quarter': df[time_col].dt.quarter,
            'dayofyear': df[time_col].dt.dayofyear,
            'weekofyear': df[time_col].dt.isocalendar().week.astype(int)
        }
    except:
        # 兼容不同pandas版本
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
    
    if verbose:
        print(f"   ✓ 已生成 {len(time_features)} 个时间特征")
    
    # 2.5 生成机器级别统计特征
    if verbose:
        print(f"\n2.5 机器级别统计特征")
    
    machine_stats = df.groupby(machine_col)[target_col].agg([
        'mean', 'std', 'min', 'max', 'median'
    ]).reset_index()
    
    machine_stats.columns = [machine_col, 
                            f'{target_col}_machine_mean', 
                            f'{target_col}_machine_std', 
                            f'{target_col}_machine_min', 
                            f'{target_col}_machine_max',
                            f'{target_col}_machine_median']
    
    df = df.merge(machine_stats, on=machine_col, how='left')
    machine_feature_cols = [col for col in machine_stats.columns if col != machine_col]
    feature_cols.extend(machine_feature_cols)
    
    if verbose:
        print(f"   ✓ 已生成 {len(machine_feature_cols)} 个机器级别特征")
    
    # ==========================================
    # 步骤3: 处理特征列的缺失值（安全措施）
    # ==========================================
    stats['feature_nan_before'] = df[feature_cols].isna().sum().sum()
    
    if stats['feature_nan_before'] > 0:
        if verbose:
            print(f"\n{'='*70}")
            print(f"步骤3: 处理特征列的缺失值（安全措施）")
            print(f"{'='*70}")
            print(f"特征列缺失值总数: {stats['feature_nan_before']}")
            print(f"处理方法: {feature_fill_na_method}")
        
        df = _fill_missing_values(
            df=df,
            columns=feature_cols,
            method=feature_fill_na_method,
            group_col=machine_col if feature_fill_by_group else None,
            verbose=verbose
        )
        
        stats['feature_nan_after'] = df[feature_cols].isna().sum().sum()
        
        # 最终清理：删除仍然有缺失值的行
        if stats['feature_nan_after'] > 0:
            before_drop = len(df)
            df = df.dropna(subset=feature_cols)
            if verbose:
                print(f"   ⚠️  最终清理: 删除了 {before_drop - len(df)} 行仍有缺失值的数据")
    
    stats['final_rows'] = len(df)
    
    # 输出最终统计信息
    if verbose:
        print(f"\n{'='*70}")
        print("特征生成完成！")
        print(f"{'='*70}")
        print(f"原始数据行数: {stats['original_rows']}")
        print(f"最终数据行数: {stats['final_rows']}")
        print(f"数据保留率: {stats['final_rows']/stats['original_rows']*100:.1f}%")
        print(f"目标变量缺失值处理: {stats['target_nan_before']} → {stats['target_nan_after']}")
        print(f"生成特征总数: {len(feature_cols)}")
        print(f"特征列缺失值: {stats['feature_nan_after']}")
        print("=" * 70)
    
    return df, stats


def _fill_missing_values(
    df: pd.DataFrame,
    columns: List[str],
    method: str,
    group_col: Optional[str] = None,
    fill_value: Optional[float] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    统一的缺失值填充函数
    """
    nan_before = df[columns].isna().sum().sum()
    
    if nan_before == 0:
        return df
    
    if verbose:
        print(f"   缺失值数量: {nan_before}")
    
    if method == 'drop':
        df = df.dropna(subset=columns)
        if verbose:
            print(f"   删除含缺失值行后剩余: {len(df)} 行")
            
    elif method == 'ffill':
        if group_col:
            df[columns] = df.groupby(group_col)[columns].transform(lambda x: x.ffill())
        else:
            df[columns] = df[columns].ffill()
            
    elif method == 'bfill':
        if group_col:
            df[columns] = df.groupby(group_col)[columns].transform(lambda x: x.bfill())
        else:
            df[columns] = df[columns].bfill()
            
    elif method == 'zero':
        df[columns] = df[columns].fillna(0)
        
    elif method == 'mean':
        if group_col:
            df[columns] = df[columns].fillna(
                df.groupby(group_col)[columns].transform('mean')
            )
        else:
            df[columns] = df[columns].fillna(df[columns].mean())
            
    elif method == 'median':
        if group_col:
            df[columns] = df[columns].fillna(
                df.groupby(group_col)[columns].transform('median')
            )
        else:
            df[columns] = df[columns].fillna(df[columns].median())
            
    elif method == 'value':
        df[columns] = df[columns].fillna(fill_value if fill_value is not None else 0)
        
    elif method == 'interpolate':
        if group_col:
            df[columns] = df.groupby(group_col)[columns].transform(
                lambda x: x.interpolate(method='linear', limit_direction='both')
            )
        else:
            df[columns] = df[columns].interpolate(method='linear', limit_direction='both')
        
        # 插值后可能仍有NaN（首尾），用ffill/bfill处理
        if df[columns].isna().any().any():
            if group_col:
                df[columns] = df.groupby(group_col)[columns].transform(
                    lambda x: x.ffill().bfill()
                )
            else:
                df[columns] = df[columns].ffill().bfill()
                
    elif method == 'ffill_bfill':
        if group_col:
            df[columns] = df.groupby(group_col)[columns].transform(lambda x: x.ffill())
            df[columns] = df.groupby(group_col)[columns].transform(lambda x: x.bfill())
        else:
            df[columns] = df[columns].ffill().bfill()
            
    elif method == 'bfill_ffill':
        if group_col:
            df[columns] = df.groupby(group_col)[columns].transform(lambda x: x.bfill())
            df[columns] = df.groupby(group_col)[columns].transform(lambda x: x.ffill())
        else:
            df[columns] = df[columns].bfill().ffill()
    
    else:
        raise ValueError(f"不支持的填充方法: {method}")
    
    nan_after = df[columns].isna().sum().sum()
    if verbose and nan_after > 0:
        print(f"   处理后仍有 {nan_after} 个缺失值，用0填充")
        df[columns] = df[columns].fillna(0)
    
    return df


def demo_comparison():
    """
    对比在特征生成前/后处理缺失值的效果
    """
    np.random.seed(42)
    
    # 创建包含缺失值的数据
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    data_list = []
    for machine_id in [1, 2, 3]:
        for date in dates:
            base_value = machine_id * 10
            noise = np.random.normal(0, 5)
            value = base_value + np.sin(date.day / 10) * 5 + noise
            
            # 20%缺失率
            if np.random.random() < 0.2:
                value = np.nan
                
            data_list.append({
                'machine_id': machine_id,
                'date': date,
                'temperature': value
            })
    
    df = pd.DataFrame(data_list)
    
    print("=" * 80)
    print("对比：特征生成前 vs 特征生成后 处理缺失值")
    print("=" * 80)
    print(f"原始数据: {len(df)} 行, "
          f"目标变量缺失: {df['temperature'].isna().sum()} 个")
    
    # 方法1: 在特征生成前填补（推荐）
    print(f"\n{'='*80}")
    print("方法1: 在特征生成前用插值填补目标变量（推荐）")
    print('='*80)
    df1, stats1 = generate_features(
        data=df,
        machine_col='machine_id',
        time_col='date',
        target_col='temperature',
        lag_periods=[1, 3, 7],
        rolling_windows=[7, 14],
        target_fill_na_method='interpolate',  # 先生成前填补目标变量
        target_fill_by_group=True,
        feature_fill_na_method='ffill',  # 特征列的备用处理
        verbose=True
    )
    print(df1.head())
    # 方法2: 在特征生成后填补（不推荐）
    print(f"\n{'='*80}")
    print("方法2: 在特征生成后用前向填充处理所有特征（不推荐）")
    print('='*80)
    df2, stats2 = generate_features(
        data=df,
        machine_col='machine_id',
        time_col='date',
        target_col='temperature',
        lag_periods=[1, 3, 7],
        rolling_windows=[7, 14],
        target_fill_na_method='drop',  # 先不做任何处理（使用drop实际上是跳过处理）
        target_fill_by_group=True,
        feature_fill_na_method='ffill',  # 在特征生成后处理
        verbose=True
    )
    print(df2.head())
    # 对比结果
    print(f"\n{'='*80}")
    print("结果对比")
    print('='*80)
    print(f"{'指标':<30} {'方法1(前处理)':<20} {'方法2(后处理)':<20}")
    print("-" * 70)
    print(f"{'原始数据行数':<30} {stats1['original_rows']:<20} {stats2['original_rows']:<20}")
    print(f"{'最终数据行数':<30} {stats1['final_rows']:<20} {stats2['final_rows']:<20}")
    print(f"{'数据保留率':<30} {stats1['final_rows']/stats1['original_rows']*100:.1f}%{'':<16} "
          f"{stats2['final_rows']/stats2['original_rows']*100:.1f}%")
    print(f"{'目标变量缺失(处理前)':<30} {stats1['target_nan_before']:<20} {stats2['target_nan_before']:<20}")
    print(f"{'目标变量缺失(处理后)':<30} {stats1['target_nan_after']:<20} {stats2['target_nan_after']:<20}")
    print(f"{'特征列缺失值':<30} {stats1['feature_nan_after']:<20} {stats2['feature_nan_after']:<20}")
    
    print(f"\n✅ 推荐使用方法1（特征生成前处理缺失值）")
    print("   优点: 数据保留率高，特征质量更好，统计量更准确")
    
    return df1, df2, stats1, stats2


if __name__ == "__main__":
    try:
        df1, df2, stats1, stats2 = demo_comparison()
        print("\n✅ 演示完成！")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()