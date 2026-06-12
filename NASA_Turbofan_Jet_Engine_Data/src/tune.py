"""
============================================================================
NASA CMAPSS FD001 — XGBoost 超参数调优
============================================================================

四种主流调参方法对比:

┌─────────────────┬──────────────────────┬──────────┬──────────────────────┐
│ 方法              │ 原理                   │ 效率       │ 适用场景               │
├─────────────────┼──────────────────────┼──────────┼──────────────────────┤
│ 1. Grid Search  │ 穷举所有参数组合          │ ★☆☆☆☆    │ 参数少(<3)时，了解交互   │
│ 2. Random Search│ 随机采样参数空间          │ ★★☆☆☆    │ 高维空间比Grid有效      │
│ 3. Bayesian Opt │ 用概率模型(TPE/GP)引导搜索│ ★★★★☆    │ **推荐** 工业界标配     │
│ 4. Hyperband    │ 早停差试验 + 渐进加大资源  │ ★★★★★    │ 超大搜索空间+预算有限   │
└─────────────────┴──────────────────────┴──────────┴──────────────────────┘

方法详解:

1. Grid Search (网格搜索)
   - 原理: 对每个参数预设几个候选值，穷举所有组合
   - 缺陷: 维度灾难 — 5个参数各5个值 = 3125种组合，很多组合是浪费的
   - 适用: 只调2-3个关键参数时的快速摸底

2. Random Search (随机搜索)
   - 原理: 从参数分布中随机采样N组
   - 依据: Bergstra & Bengio (2012) 证明 — 高维空间中随机搜索比网格搜索
          更高效，因为大多数参数对结果影响不大，网格把预算浪费在无关参数上
   - 优势: 可设定预算(N次试验)，覆盖更广的参数空间

3. Bayesian Optimization (贝叶斯优化) ← 本脚本使用的方法
   - 原理: 构建 surrogate model (高斯过程/TPE) 来近似 f(params) → score，
          每次选"最可能改进当前最优"的参数组合。核心公式:
          EI(x) = E[max(f(x) - y_best, 0)]  (Expected Improvement)
   - TPE (Tree-structured Parzen Estimator): Optuna 默认算法，
          用两个核密度估计 l(x) 和 g(x) 分别拟合"好试验"和"差试验"的分布，
          采样时最大化 l(x)/g(x) 比值
   - 依据: 相比 Random Search，贝叶斯优化用更少的试验找到更优解
          (通常在 50-100 次试验内收敛到接近最优)
   - 优势: 自动平衡 exploration(探索未知区域) vs exploitation(深耕已知好区域)

4. Hyperband
   - 原理: 大量启动廉价试验(少树/少数据)，逐步淘汰差的，只给好的加资源
   - 依据: Successive Halving — 很多参数组合早期就能看出不行
   - 适用: 搜索空间非常大(10+参数)，预算有限

XGBoost 参数调优顺序 (按影响力从大到小):

 第一梯队 (最大影响 — 先调):
   n_estimators    : 树的数量，配合 early_stopping 自动确定
   learning_rate   : 学习率，与 n_estimators 反向关系
   max_depth       : 树深度，控制模型复杂度

 第二梯队 (正则化 — 防过拟合):
   subsample       : 行采样率
   colsample_bytree: 列采样率 (特征多时尤其重要，本任务381维)
   min_child_weight: 叶子最小样本权重和

 第三梯队 (微调):
   reg_alpha       : L1 正则
   reg_lambda      : L2 正则
   gamma           : 分裂所需最小损失减少量

调参策略:
  - 先用默认参数跑 baseline
  - 粗调第一梯队(wide range) → 细调第二梯队(narrow range)
  - 最后微调第三梯队
  - 总共 50-100 次试验通常足够

运行:
    python tune.py          # 默认100次试验
    python tune.py 50       # 50次快速调参
"""

import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 复用特征工程模块
from feature_engineering import build_features

# ============================================================================
# 0. 配置
# ============================================================================
DATA_DIR = '../raw/CMaps'
COL_NAMES = (
    ['unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
    + [f's_{i}' for i in range(1, 22)]
)
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 100  # 命令行可指定试验次数
RUL_CLIP = 125


# ============================================================================
# 1. 加载数据 & 构建特征（只跑一次，所有试验共享）
# ============================================================================
def load_and_build_features():
    """加载数据并构建特征矩阵（只执行一次，避免重复计算）"""
    print('加载数据 & 构建特征...')

    df_train = pd.read_csv(f'{DATA_DIR}/train_FD001.txt', sep=r'\s+',
                           header=None, names=COL_NAMES)
    df_valid = pd.read_csv(f'{DATA_DIR}/test_FD001.txt', sep=r'\s+',
                           header=None, names=COL_NAMES)
    y_valid_raw = pd.read_csv(f'{DATA_DIR}/RUL_FD001.txt', sep=r'\s+',
                              header=None, names=['RUL'])

    # 特征工程
    X_train, y_train, regime_means = build_features(df_train, is_train=True)
    X_valid_all = build_features(df_valid, is_train=False,
                                 train_regime_means=regime_means)

    # Valid: 仅取每个发动机最后一个 cycle
    last_idx = df_valid.groupby('unit_number')['time_cycles'].idxmax().values
    X_valid = X_valid_all.loc[last_idx].reset_index(drop=True)
    y_valid = y_valid_raw['RUL'].clip(upper=RUL_CLIP).values.ravel()

    print(f'  X_train: {X_train.shape}, X_valid: {X_valid.shape}')
    return X_train, y_train, X_valid, y_valid


# ============================================================================
# 2. 定义 Optuna 目标函数
# ============================================================================
def objective(trial, X_train, y_train, X_valid, y_valid):
    """
    Optuna 目标函数 —— 每次试验训练一个 XGBoost 并返回 valid RMSE。

    Optuna 使用 TPE (Tree-structured Parzen Estimator) 算法:
      - 将历史试验分为"好"(前γ%)和"差"(后1-γ%)两组
      - 用核密度估计拟合两组分布: l(x|y好) 和 g(x|y差)
      - 采样时最大化 l(x)/g(x) → 选"更像好的、不像差的"参数
      - 自动平衡 exploration/exploitation
    """
    # --- 第一梯队: 核心结构参数 ---
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    n_estimators = trial.suggest_int('n_estimators', 100, 500)

    # --- 第二梯队: 正则化参数 ---
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)

    # --- 第三梯队: L1/L2 正则 ---
    reg_alpha = trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True)
    reg_lambda = trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42,
        n_jobs=-1,
        # xgboost 3.x 要求 early_stopping 在构造中设置
        early_stopping_rounds=30,
        eval_metric='rmse',
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    # 取最佳迭代对应的 RMSE
    best_rmse = model.evals_result()['validation_0']['rmse'][model.best_iteration]
    return best_rmse


# ============================================================================
# 3. 运行调优
# ============================================================================
def main():
    print('=' * 60)
    print('XGBoost 超参数调优 — Optuna (Bayesian Optimization)')
    print('=' * 60)
    print(f'试验次数: {N_TRIALS}')
    print(f'算法: TPE (Tree-structured Parzen Estimator)')
    print()

    # 加载数据（只一次）
    X_train, y_train, X_valid, y_valid = load_and_build_features()

    # 创建 study，方向=最小化 RMSE
    # pruner: 提前终止明显没希望的试验（MedianPruner 是默认）
    study = optuna.create_study(
        direction='minimize',
        study_name='xgboost_cmapss',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,    # 前5次不剪枝（建立baseline）
            n_warmup_steps=10,     # 每个trial前10步不剪枝
        ),
    )

    # 运行优化
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_valid, y_valid),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    # ======================================================================
    # 4. 结果分析
    # ======================================================================
    print('\n' + '=' * 60)
    print('调优结果')
    print('=' * 60)

    print(f'\nBest trial (#{study.best_trial.number}): RMSE = {study.best_value:.2f}')
    print(f'\nBest params:')
    for k, v in study.best_params.items():
        print(f'  {k:<20} = {v}')

    # 参数重要性: Optuna 分析每个参数对结果的贡献
    print(f'\n参数重要性 (越高 = 该参数对RMSE影响越大):')
    importances = optuna.importance.get_param_importances(study)
    for param, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = '#' * int(imp * 40)
        print(f'  {param:<20} {imp:.3f}  {bar}')

    # 对比: baseline vs tuned
    print(f'\n{"="*60}')
    print(f'性能对比')
    print(f'{"="*60}')

    # Baseline: 使用经验默认参数
    baseline_params = {
        'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 300,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
    }
    baseline = xgb.XGBRegressor(
        objective='reg:squarederror', random_state=42, n_jobs=-1,
        early_stopping_rounds=30, eval_metric='rmse', **baseline_params,
    )
    baseline.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    baseline_rmse = min(baseline.evals_result()['validation_0']['rmse'])

    # Tuned
    tuned_params = study.best_params
    tuned_params.update({
        'objective': 'reg:squarederror', 'random_state': 42, 'n_jobs': -1,
        'early_stopping_rounds': 30, 'eval_metric': 'rmse',
    })
    tuned = xgb.XGBRegressor(**tuned_params)
    tuned.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    tuned_rmse = min(tuned.evals_result()['validation_0']['rmse'])

    improvement = (baseline_rmse - tuned_rmse) / baseline_rmse * 100
    print(f'  Baseline RMSE: {baseline_rmse:.2f}')
    print(f'  Tuned RMSE:    {tuned_rmse:.2f}')
    print(f'  Improvement:   {improvement:.1f}%')

    # 最后用最佳参数 + eval_set=(X_train, X_valid) 完整训练
    # 用于最终提交
    print(f'\n最终模型参数 (可直接复制到 train.py 的 XGB_PARAMS):')
    print(f'XGB_PARAMS = {{')
    for k, v in study.best_params.items():
        print(f"    '{k}': {v!r},")
    print(f"    'objective': 'reg:squarederror',")
    print(f"    'random_state': 42,")
    print(f"    'n_jobs': -1,")
    print(f"    'early_stopping_rounds': 30,")
    print(f"    'eval_metric': 'rmse',")
    print(f'}}')

    # 保存 study 结果（可后续用 optuna.visualization 绘图）
    # study.trials_dataframe().to_csv('tuning_history.csv', index=False)
    # print('\nTuning history saved to tuning_history.csv')


if __name__ == '__main__':
    main()
