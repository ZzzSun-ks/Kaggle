"""
============================================================================
NASA CMAPSS Turbofan Engine — 特征工程模块
============================================================================

基于 _draft.ipynb 中 7 个阶段 EDA 的发现，设计并实现 5 类特征：

EDA 发现                              → 特征工程对策
──────────────────────────────────────────────────────────────
1. 原始传感器噪声大（RUL对齐图 ±1σ 带很宽） → 滑动窗口统计量（去噪）
2. 退化幅度 <2% 但方向一致              → 窗口线性趋势斜率（方向 > 绝对值）
3. RUL<50 时部分传感器斜率增大           → 短/长窗口变化率比值（捕捉加速）
4. setting 每 cycle 都变，100% 发动机有工况切换 → 工况残差（去工况效应）
5. 传感器聚为两大对抗集团                → 集团比值（放大退化信号）

传感器聚类结果（距离阈值 0.5）：
  Cluster 1 (2个): s_9, s_14          — 独立集团，与RUL相关性弱
  Cluster 2 (12个): s_2, s_3, s_4, s_7, s_8, s_11, s_12,
                     s_13, s_15, s_17, s_20, s_21  — 核心集团
  Cluster 3 (1个): s_6                — 极低方差，弱信号

PCA 结果：
  前 7 个主成分解释 90% 方差 → 有效维度约为原始的 47%

剔除的零/低方差特征（基于 describe().T 分析）：
  s_1, s_5, s_10, s_16, s_18, s_19: 零方差（完全无信息）
  s_6: 极低方差（std=0.0014）
  setting_3: 零方差（恒定值 100）

使用方法：
    from feature_engineering import build_features
    X_train, y_train = build_features(train_df, is_train=True)
    X_valid, _ = build_features(valid_df, is_train=False,
                                 train_for_regime=train_df)
"""

import numpy as np
import pandas as pd

# ============================================================================
# 全局配置 — 基于 EDA 结论
# ============================================================================

# 有效传感器（排除零方差/极低方差的 7 个死传感器 + s_6）
FEATURE_SENSORS = [
    's_2', 's_3', 's_4',  's_7',  's_8', 's_9',
    's_11', 's_12', 's_13', 's_14', 's_15', 's_17',
    's_20', 's_21'
]

# 滑动窗口大小（cycles）
WINDOWS = [5, 10, 20, 30]

# RUL 截断阈值（CMAPSS 文献标准值：125~130）
RUL_CLIP = 125

# 传感器集团定义（基于 RUL 相关性方向）
# Group A: 与 RUL 正相关 → 故障前数值下降
GROUP_RUL_POS = ['s_7', 's_12', 's_20', 's_21']
# Group B: 与 RUL 负相关 → 故障前数值上升
GROUP_RUL_NEG = ['s_2', 's_3', 's_4', 's_11', 's_15', 's_17']

# 要保留的原始列（传感器 + 工况）
BASE_COLUMNS = FEATURE_SENSORS + ['setting_1', 'setting_2']


# ============================================================================
# 特征 A: 滑动窗口统计量
# 依据: RUL 对齐图中 ±1σ 带很宽 → 原始信号噪声大，滚动统计可以平滑噪声、
#       捕获局部退化趋势
# 产出: mean, std, min, max → 每个传感器 × 每个窗口 = 4 stats/sensor/window
# ============================================================================
def make_rolling_features(df, sensors, windows):
    """为每个传感器计算滑动窗口统计量（mean, std, min, max）"""
    features = {}
    for sensor in sensors:
        for w in windows:
            roll = df.groupby('unit_number')[sensor].rolling(
                window=w, min_periods=1
            )
            features[f'{sensor}_roll_mean_{w}'] = roll.mean().reset_index(level=0, drop=True)
            features[f'{sensor}_roll_std_{w}']  = roll.std().reset_index(level=0, drop=True)
            features[f'{sensor}_roll_min_{w}']  = roll.min().reset_index(level=0, drop=True)
            features[f'{sensor}_roll_max_{w}']  = roll.max().reset_index(level=0, drop=True)
    return pd.DataFrame(features)


# ============================================================================
# 特征 B: 窗口线性趋势（斜率）
# 依据: 退化幅度 <2% 但方向一致 → 对于预测性维护，"传感器在往哪个方向走"
#       比"传感器当前值是多少"更重要
# 产出: 每个传感器 × 每个窗口 = 1 slope/sensor/window
# ============================================================================
def make_trend_features(df, sensors, windows):
    """计算滑动窗口内的线性趋势斜率"""
    features = {}
    for sensor in sensors:
        for w in windows:

            def slope_over_window(x):
                if len(x) < 2:
                    return 0.0
                t = np.arange(len(x))
                slope, _ = np.polyfit(t, x, 1)
                return slope

            trend = df.groupby('unit_number')[sensor].rolling(
                window=w, min_periods=2
            ).apply(slope_over_window, raw=True)
            features[f'{sensor}_trend_{w}'] = trend.reset_index(level=0, drop=True)
    return pd.DataFrame(features)


# ============================================================================
# 特征 C: 变化率 & 加速退化信号
# 依据: RUL 对齐图中部分传感器在 RUL<50 时曲线斜率增大 →
#       短窗口 vs 长窗口的变化率可以捕捉"加速退化"
# 产出: 5v20 变化率, 5v30 变化率, 绝对加速度
# ============================================================================
def make_rate_features(df, sensors):
    """计算短窗口 vs 长窗口的变化率，捕捉加速退化"""
    features = {}
    for sensor in sensors:
        roll_5  = df.groupby('unit_number')[sensor].rolling(5,  min_periods=1).mean().reset_index(level=0, drop=True)
        roll_20 = df.groupby('unit_number')[sensor].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
        roll_30 = df.groupby('unit_number')[sensor].rolling(30, min_periods=1).mean().reset_index(level=0, drop=True)

        # 短期 vs 长期变化率: 正值 = 近期上升, 负值 = 近期下降
        features[f'{sensor}_rate_5v20'] = (roll_5 - roll_20) / (roll_20.abs() + 1e-8)
        features[f'{sensor}_rate_5v30'] = (roll_5 - roll_30) / (roll_30.abs() + 1e-8)

        # 变化加速度: 近期趋势 - 远期趋势（绝对差值）
        features[f'{sensor}_acc_5v20'] = roll_5 - roll_20
    return pd.DataFrame(features)


# ============================================================================
# 特征 D: 工况归一化残差
# 依据: EDA 发现 setting_1, setting_2 每 cycle 都变化（100/100 台发动机
#       都有工况变化），且共有 85 种不同组合。传感器读数同时受退化 + 工况
#       双重影响。去除工况效应后的残差才是"纯退化信号"。
# 方法: 将连续 setting 分箱为 10×10 网格，每个网格内计算传感器全局均值
#       作为"正常基准"，当前值减去该基准得到残差。
# 产出: 每个传感器 1 个残差特征 = 15 个残差
# ============================================================================
def make_regime_residual_features(df, sensors, regime_means=None):
    """计算传感器相对于其当前工况下均值的偏离（残差）

    Args:
        df: 输入数据
        sensors: 传感器列表
        regime_means: 如果提供，使用该预计算的工况均值（用于valid集）
    """
    df = df.copy()
    df['s1_bin'] = pd.cut(df['setting_1'], bins=10, labels=False)
    df['s2_bin'] = pd.cut(df['setting_2'], bins=10, labels=False)
    df['regime_bin'] = df['s1_bin'].astype(str) + '_' + df['s2_bin'].astype(str)

    if regime_means is not None:
        # 使用训练集的工况均值（避免数据泄露）
        for sensor in sensors:
            df[f'{sensor}_regime_resid'] = (
                df[sensor] - df['regime_bin'].map(regime_means[sensor])
            )
        features = {f'{s}_regime_resid': df[f'{s}_regime_resid'] for s in sensors}
    else:
        # 计算当前数据集的工况均值
        means = df.groupby('regime_bin')[sensors].transform('mean')
        features = {}
        for sensor in sensors:
            features[f'{sensor}_regime_resid'] = df[sensor] - means[sensor]

    return pd.DataFrame(features), None if regime_means is not None else _compute_regime_means(df, sensors)


def _compute_regime_means(df, sensors):
    """计算每个工况分箱下各传感器的均值，供 valid 集复用"""
    means = {}
    for s in sensors:
        means[s] = df.groupby('regime_bin')[s].mean()
    return means


# ============================================================================
# 特征 E: 传感器集团比值
# 依据: 聚类发现传感器分为两大对抗集团——
#       Group A (正相关 RUL): s_7, s_12, s_20, s_21 — 故障前下降
#       Group B (负相关 RUL): s_2, s_3, s_4, s_11, s_15, s_17 — 故障前上升
#       两个集团的比值会放大退化信号（分子下降、分母上升 → 比值加速下降）
#       EDA 验证: ratio_pos_neg 与 RUL 的 r = 0.727，远超任一单独传感器
# 产出: group_pos_mean, group_neg_mean, ratio_pos_neg, log_ratio_pos_neg
# ============================================================================
def make_ratio_features(df):
    """计算传感器集团间的比值特征"""
    features = pd.DataFrame()
    features['group_pos_mean'] = df[GROUP_RUL_POS].mean(axis=1)
    features['group_neg_mean'] = df[GROUP_RUL_NEG].mean(axis=1)
    # 比值 — 退化时分子下降、分母上升，比值加速下降
    features['ratio_pos_neg'] = (
        features['group_pos_mean'] / (features['group_neg_mean'] + 1e-8)
    )
    # 对数比值 — 数值分布更稳定，避免极端值
    features['log_ratio_pos_neg'] = (
        np.log(features['group_pos_mean'] + 1e-8)
        - np.log(features['group_neg_mean'] + 1e-8)
    )
    return features


# ============================================================================
# 主入口: 完整特征工程管线
# ============================================================================
def build_features(df, is_train=True, train_regime_means=None):
    """对输入 DataFrame 执行完整的特征工程管线

    Args:
        df: 输入数据，必须包含 unit_number, time_cycles, setting_1, setting_2
            以及 FEATURE_SENSORS 中的所有传感器列
        is_train: 是否为训练集（决定是否生成 RUL_clipped 标签）
        train_regime_means: 训练集的工况均值（用于 valid 集，避免数据泄露）

    Returns:
        X: 特征矩阵 (DataFrame)
        y: 目标变量 (Series, 仅 is_train=True 时返回)
    """
    # --- 基础特征（原始传感器 + setting） ---
    base = df[BASE_COLUMNS].copy()
    base.columns = [f'{c}_raw' if c in FEATURE_SENSORS else c for c in base.columns]

    # --- A: 滑动窗口统计量 ---
    rolling_feats = make_rolling_features(df, FEATURE_SENSORS, WINDOWS)

    # --- B: 窗口线性趋势 ---
    trend_feats = make_trend_features(df, FEATURE_SENSORS, WINDOWS)

    # --- C: 变化率/加速 ---
    rate_feats = make_rate_features(df, FEATURE_SENSORS)

    # --- D: 工况残差 ---
    if is_train:
        regime_feats, regime_means = make_regime_residual_features(df, FEATURE_SENSORS)
    else:
        regime_feats, _ = make_regime_residual_features(
            df, FEATURE_SENSORS, regime_means=train_regime_means
        )

    # --- E: 集团比值 ---
    ratio_feats = make_ratio_features(df)

    # --- 拼接所有特征 ---
    X = pd.concat([
        base, rolling_feats, trend_feats, rate_feats, regime_feats, ratio_feats
    ], axis=1)

    # --- 处理无效值 ---
    X = X.bfill().ffill().fillna(0)

    if is_train:
        # 生成标签：RUL 并截断
        max_cycles = df.groupby('unit_number')['time_cycles'].transform('max')
        rul = max_cycles - df['time_cycles']
        y = rul.clip(upper=RUL_CLIP)
        return X, y, regime_means
    else:
        return X


# ============================================================================
# 特征维度统计（供参考）
# ============================================================================
if __name__ == '__main__':
    n_sensors = len(FEATURE_SENSORS)
    print('=' * 60)
    print('NASA CMAPSS 特征工程模块')
    print('=' * 60)
    print(f'')
    print(f'有效传感器: {n_sensors} 个')
    print(f'滑动窗口:   {WINDOWS}')
    print(f'RUL 截断:   {RUL_CLIP}')
    print(f'')
    print(f'特征维度明细:')
    print(f'  A. 滑动窗口统计: {n_sensors} sensors x {len(WINDOWS)} windows x 4 stats = {n_sensors * len(WINDOWS) * 4}')
    print(f'  B. 窗口趋势:     {n_sensors} sensors x {len(WINDOWS)} windows = {n_sensors * len(WINDOWS)}')
    print(f'  C. 变化率/加速:  {n_sensors} sensors x 3 ratios = {n_sensors * 3}')
    print(f'  D. 工况残差:      {n_sensors} sensors = {n_sensors}')
    print(f'  E. 集团比值:      4')
    print(f'  + 原始传感器:     {len(BASE_COLUMNS)}')
    print(f'  ─────────────────────')
    total = n_sensors * len(WINDOWS) * 4 + n_sensors * len(WINDOWS) + n_sensors * 3 + n_sensors + 4 + len(BASE_COLUMNS)
    print(f'  总计:             {total}')
