"""
============================================================================
NASA CMAPSS Turbofan Engine — XGBoost 训练与评估
============================================================================

基于 feature_engineering.py 构建的特征矩阵，训练 XGBoost 回归模型
预测发动机剩余使用寿命 (RUL, Remaining Useful Life)。

数据:
  - train_FD001.txt: 100 台发动机的运行至故障数据
  - test_FD001.txt:  100 台发动机的运行数据（故障前截断）
  - RUL_FD001.txt:   测试集每台发动机的真实 RUL（仅最后1个cycle）

评估: CMAPSS 标准使用每个发动机最后一个 cycle 的预测 RMSE

运行:
    python train.py

配置:
    修改下方 XGB_PARAMS 和 TRAIN_PARAMS 字典来调整模型
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 导入特征工程模块
from feature_engineering import (
    build_features, FEATURE_SENSORS, BASE_COLUMNS,
    RUL_CLIP, WINDOWS, GROUP_RUL_POS, GROUP_RUL_NEG
)

# ============================================================================
# 0. 配置
# ============================================================================

DATA_DIR = '../raw/CMaps'

# 列名定义
INDEX_NAMES   = ['unit_number', 'time_cycles']
SETTING_NAMES = ['setting_1', 'setting_2', 'setting_3']
SENSOR_NAMES  = [f's_{i}' for i in range(1, 22)]
COL_NAMES     = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES

# XGBoost 参数
# 说明:
#   objective='reg:squarederror': 回归 → 最小化 MSE，对应 RMSE 评估
#   n_estimators=300: 足够多的树捕捉非线性退化模式
#   learning_rate=0.05: 小学习率 + early_stopping 防过拟合
#   max_depth=6: 适中深度，退化信号弱不宜太深
#   subsample/colsample_bytree=0.8: 随机采样 → 类似 Bagging，增加泛化
#   min_child_weight=3: 叶子节点最小样本权重，防过拟合
#   reg_alpha=0.1: L1 正则 → 自动压制噪声特征（381维中有不少弱信号）
#   reg_lambda=1.0: L2 正则
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    # xgboost 3.x: early_stopping_rounds 和 eval_metric 在构造函数传
    'early_stopping_rounds': 30,
    'eval_metric': 'rmse',
}

# 训练参数 (xgboost 3.x: early_stopping_rounds/eval_metric 已移到 XGB_PARAMS 构造中)
TRAIN_PARAMS = {
    'verbose': 50,  # 每 50 轮输出一次
}


# ============================================================================
# 1. 加载数据
# ============================================================================
def load_data():
    """加载 CMAPSS FD001 数据集"""
    print('=' * 60)
    print('1. 加载数据')
    print('=' * 60)

    df_train = pd.read_csv(
        f'{DATA_DIR}/train_FD001.txt', sep=r'\s+',
        header=None, names=COL_NAMES
    )
    df_valid = pd.read_csv(
        f'{DATA_DIR}/test_FD001.txt', sep=r'\s+',
        header=None, names=COL_NAMES
    )
    y_valid_raw = pd.read_csv(
        f'{DATA_DIR}/RUL_FD001.txt', sep=r'\s+',
        header=None, names=['RUL']
    )

    print(f'  Train: {df_train.shape[0]} 条 x {df_train.shape[1]} 列')
    print(f'  Valid: {df_valid.shape[0]} 条 x {df_valid.shape[1]} 列')
    print(f'  y_valid: {y_valid_raw.shape[0]} 条（每引擎1个RUL）')
    return df_train, df_valid, y_valid_raw


# ============================================================================
# 2. 构建特征
# ============================================================================
def build_all_features(df_train, df_valid):
    """对 train 和 valid 执行完整的特征工程管线"""
    print('\n' + '=' * 60)
    print('2. 构建特征（滑动窗口/趋势/变化率/残差/比值）')
    print('=' * 60)

    # 训练集：生成 X_train, y_train，同时获取工况均值供 valid 复用
    X_train, y_train, regime_means = build_features(df_train, is_train=True)
    # 验证集：使用训练集的工况均值（避免数据泄露）
    X_valid_all = build_features(
        df_valid, is_train=False, train_regime_means=regime_means
    )

    print(f'  X_train: {X_train.shape[0]} 条 x {X_train.shape[1]} 维')
    print(f'  y_train: {y_train.shape[0]} 条, range [{y_train.min():.0f}, {y_train.max():.0f}]')
    print(f'  X_valid: {X_valid_all.shape[0]} 条 x {X_valid_all.shape[1]} 维')
    return X_train, y_train, X_valid_all


# ============================================================================
# 3. 准备验证集（CMAPSS 标准：仅评估最后一个 cycle）
# ============================================================================
def prepare_valid_for_eval(df_valid, X_valid_all, y_valid_raw):
    """从 valid 全量数据中提取每个发动机的最后一个 cycle

    CMAPSS 标准评估方式:
      - y_valid (RUL_FD001.txt) 只有 100 行 —— 每台发动机 1 个 RUL
      - valid/test 集有 13096 行 —— 每台发动机多个 cycle
      - 训练时 eval_set 或最终评估，必须对齐：只取最后1个cycle
    """
    print('\n' + '=' * 60)
    print('3. 准备验证集（仅取每个发动机最后1个cycle）')
    print('=' * 60)

    # 每个发动机最后一条记录的索引
    last_idx = df_valid.groupby('unit_number')['time_cycles'].idxmax().values
    X_valid_last = X_valid_all.loc[last_idx].reset_index(drop=True)

    # y_valid 已经是按发动机顺序排列的，转为一维数组
    y_valid_last = y_valid_raw['RUL'].clip(upper=RUL_CLIP).values.ravel()

    print(f'  X_valid (全部 cycle): {X_valid_all.shape}')
    print(f'  X_valid (仅最后1个):  {X_valid_last.shape}')
    print(f'  y_valid:              {y_valid_last.shape}')
    print(f'  列一致:               {all(X_valid_last.columns == X_valid_all.columns)}')
    return X_valid_last, y_valid_last


# ============================================================================
# 4. 训练 XGBoost
# ============================================================================
def train_xgboost(X_train, y_train, X_valid_last, y_valid_last):
    """训练 XGBoost 回归模型"""
    print('\n' + '=' * 60)
    print('4. 训练 XGBoost')
    print('=' * 60)

    model = xgb.XGBRegressor(**XGB_PARAMS)

    print(f'  参数: n_estimators={XGB_PARAMS["n_estimators"]}, '
          f'lr={XGB_PARAMS["learning_rate"]}, '
          f'max_depth={XGB_PARAMS["max_depth"]}')
    print(f'  early_stopping=30 rounds (via callback, xgboost 3.x)')

    # xgboost 3.x: early_stopping_rounds 和 eval_metric 在构造函数中设置
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid_last, y_valid_last)],
        **TRAIN_PARAMS,
    )

    print(f'\n  训练完成!')
    print(f'  最佳迭代: {model.best_iteration}')
    # best_score 是 eval_set 中最后一个集合的 RMSE
    evals = model.evals_result()
    best_train_rmse = min(evals['validation_0']['rmse'])
    best_valid_rmse = min(evals['validation_1']['rmse'])
    print(f'  最佳 Train RMSE: {best_train_rmse:.2f}')
    print(f'  最佳 Valid RMSE: {best_valid_rmse:.2f}  ← 用于 early stopping')
    return model


# ============================================================================
# 5. 评估 & 可视化
# ============================================================================
def evaluate_and_plot(model, X_valid_last, y_valid_last):
    """在验证集上评估并绘制诊断图"""
    print('\n' + '=' * 60)
    print('5. 验证集评估')
    print('=' * 60)

    # 预测（并限制在有效范围）
    y_pred = model.predict(X_valid_last)
    y_pred = np.clip(y_pred, 0, RUL_CLIP)

    # 计算指标
    rmse = np.sqrt(mean_squared_error(y_valid_last, y_pred))
    mae  = mean_absolute_error(y_valid_last, y_pred)
    r2   = r2_score(y_valid_last, y_pred)

    print(f'  RMSE: {rmse:.2f} cycles  ← CMAPSS FD001 标准评估指标')
    print(f'  MAE:  {mae:.2f} cycles')
    print(f'  R2:   {r2:.4f}')

    # --- 图1: 预测 vs 真实 + 误差分布 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(y_valid_last, y_pred, alpha=0.5, s=20, color='steelblue')
    ax.plot([0, RUL_CLIP], [0, RUL_CLIP], 'r--', linewidth=1.5, label='Perfect')
    ax.set_xlabel('True RUL (clipped)')
    ax.set_ylabel('Predicted RUL')
    ax.set_title(f'Predicted vs True (RMSE={rmse:.1f})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    errors = y_pred - y_valid_last
    ax2.hist(errors, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax2.axvline(x=errors.mean(), color='orange', linestyle='--', linewidth=1.5,
                label=f'Bias={errors.mean():.1f}')
    ax2.set_xlabel('Prediction Error (Predicted - True)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Error Distribution (std={errors.std():.1f})', fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('  图表已保存: evaluation.png')

    # --- 图2: 学习曲线 ---
    evals = model.evals_result()
    fig2, ax = plt.subplots(figsize=(10, 5))
    ax.plot(evals['validation_0']['rmse'], label='Train RMSE', alpha=0.7)
    ax.plot(evals['validation_1']['rmse'], label='Valid RMSE', alpha=0.7)
    ax.axvline(x=model.best_iteration, color='red', linestyle='--',
               label=f'Best iter={model.best_iteration}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('RMSE')
    ax.set_title('XGBoost Learning Curve', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('  图表已保存: learning_curve.png')


# ============================================================================
# 6. 特征重要性分析
# ============================================================================
def feature_importance_analysis(model, X_train):
    """分析特征重要性，验证特征工程设计的有效性"""
    print('\n' + '=' * 60)
    print('6. 特征重要性分析')
    print('=' * 60)

    importance = pd.DataFrame({
        'feature': X_train.columns,
        'gain': model.feature_importances_,
    }).sort_values('gain', ascending=False)

    # Top 20
    print('\n  Top 20 特征 (按 Gain):')
    print(f'  {"Rank":<5} {"类别":<12} {"特征名":<50} {"Gain":>8}')
    print(f'  {"─"*5} {"─"*12} {"─"*50} {"─"*8}')
    for i, (_, row) in enumerate(importance.head(20).iterrows()):
        fname = row['feature']
        if '_roll_' in fname:     tag = '滑动统计'
        elif '_trend_' in fname:  tag = '趋势'
        elif '_rate_' in fname or '_acc_' in fname: tag = '变化率'
        elif '_regime_resid' in fname: tag = '工况残差'
        elif 'ratio' in fname or 'group_' in fname: tag = '集团比值'
        elif '_raw' in fname:     tag = '原始值'
        elif 'setting' in fname:  tag = 'Setting'
        else:                     tag = '其他'
        print(f'  {i+1:<5} {tag:<12} {fname:<50} {row["gain"]:>8.4f}')

    # 各类特征贡献占比
    tag_groups = {
        '滑动窗口统计': importance[importance['feature'].str.contains('_roll_')]['gain'].sum(),
        '窗口趋势':     importance[importance['feature'].str.contains('_trend_')]['gain'].sum(),
        '变化率/加速':  importance[importance['feature'].str.contains('_rate_|_acc_')]['gain'].sum(),
        '工况残差':     importance[importance['feature'].str.contains('_regime_resid')]['gain'].sum(),
        '集团比值':     importance[importance['feature'].str.contains('ratio|group_')]['gain'].sum(),
        '原始传感器':   importance[importance['feature'].str.contains('_raw')]['gain'].sum(),
        'Setting':      importance[importance['feature'].str.contains('setting')]['gain'].sum(),
    }
    total_gain = sum(tag_groups.values())
    print('\n  各类特征总重要性占比:')
    for tag, g in sorted(tag_groups.items(), key=lambda x: -x[1]):
        bar = '█' * int(g / total_gain * 40)
        print(f'    {tag:<15} {g/total_gain*100:>5.1f}%  {bar}')

    # 图3: 特征重要性 Top 20
    fig, ax = plt.subplots(figsize=(10, 8))
    top20 = importance.head(20)
    colors = []
    for f in top20['feature']:
        if '_roll_' in f:    colors.append('#2196F3')
        elif '_trend_' in f: colors.append('#4CAF50')
        elif '_rate_' in f or '_acc_' in f: colors.append('#FF9800')
        elif '_regime_resid' in f: colors.append('#9C27B0')
        elif 'ratio' in f or 'group_' in f: colors.append('#E91E63')
        elif '_raw' in f:    colors.append('#607D8B')
        elif 'setting' in f: colors.append('#795548')
        else:                colors.append('#999999')

    ax.barh(range(len(top20)), top20['gain'].values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['feature'].values[::-1], fontsize=9)
    ax.set_xlabel('Feature Importance (Gain)')
    ax.set_title('Top 20 Features', fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', label='Rolling Stats'),
        Patch(facecolor='#4CAF50', label='Trend'),
        Patch(facecolor='#FF9800', label='Rate/Accel'),
        Patch(facecolor='#9C27B0', label='Regime Residual'),
        Patch(facecolor='#E91E63', label='Group Ratio'),
        Patch(facecolor='#607D8B', label='Raw Sensor'),
        Patch(facecolor='#795548', label='Setting'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('  图表已保存: feature_importance.png')


# ============================================================================
# 主流程
# ============================================================================
def main():
    print('=' * 60)
    print('NASA CMAPSS FD001 — XGBoost RUL 预测')
    print('=' * 60)

    # 1. 加载
    df_train, df_valid, y_valid_raw = load_data()

    # 2. 特征工程
    X_train, y_train, X_valid_all = build_all_features(df_train, df_valid)

    # 3. 准备验证集（仅最后1个cycle）
    X_valid_last, y_valid_last = prepare_valid_for_eval(
        df_valid, X_valid_all, y_valid_raw
    )

    # 4. 训练
    model = train_xgboost(X_train, y_train, X_valid_last, y_valid_last)

    # 5. 评估
    evaluate_and_plot(model, X_valid_last, y_valid_last)

    # 6. 特征重要性
    feature_importance_analysis(model, X_train)

    print('\n' + '=' * 60)
    print('Done!')
    print('=' * 60)


if __name__ == '__main__':
    main()
