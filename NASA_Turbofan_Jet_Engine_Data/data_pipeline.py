"""
NASA CMAPSS — 数据加载 & 特征工程管线
========================================
加载原始 txt 数据, 计算 RUL 标签, 工程化特征。
"""
import warnings
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# ============================================================
# 工具函数
# ============================================================
ROOT = Path(__file__).parent


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw(filepath):
    """加载原始 txt 数据 (空格分隔, 无表头)"""
    cfg = load_config()
    cols = cfg["columns"]
    names = cols["index"] + cols["settings"] + cols["sensors"]
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=names)
    return df


def compute_rul(df, clip=None):
    """为每条记录计算 RUL = 同发动机最大 cycle - 当前 cycle"""
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    rul = max_cycle - df["cycle"]
    if clip:
        rul = rul.clip(upper=clip)
    return rul


# ============================================================
# 特征工程
# ============================================================
def add_rolling_features(df, sensors, windows):
    """滑动窗口统计: mean, std, min, max"""
    feats = pd.DataFrame(index=df.index)
    for s in sensors:
        grp = df.groupby("unit")[s]
        for w in windows:
            roll = grp.rolling(w, min_periods=1)
            feats[f"{s}_rmean_{w}"] = roll.mean().values
            feats[f"{s}_rstd_{w}"] = roll.std().values
            feats[f"{s}_rmin_{w}"] = roll.min().values
            feats[f"{s}_rmax_{w}"] = roll.max().values
    return feats


def add_trend_features(df, sensors, windows):
    """滑动窗口线性趋势 (斜率)"""

    def _slope(x):
        if len(x) < 2:
            return 0.0
        t = np.arange(len(x))
        return np.polyfit(t, x, 1)[0]

    feats = pd.DataFrame(index=df.index)
    for s in sensors:
        for w in windows:
            feats[f"{s}_trend_{w}"] = (
                df.groupby("unit")[s]
                .rolling(w, min_periods=2)
                .apply(_slope, raw=True)
                .values
            )
    return feats


def add_rate_features(df, sensors):
    """短期 vs 长期变化率: 捕捉加速退化"""
    feats = pd.DataFrame(index=df.index)
    for s in sensors:
        r5 = df.groupby("unit")[s].rolling(5, min_periods=1).mean().values
        r20 = df.groupby("unit")[s].rolling(20, min_periods=1).mean().values
        r30 = df.groupby("unit")[s].rolling(30, min_periods=1).mean().values
        feats[f"{s}_rate_5v20"] = (r5 - r20) / (np.abs(r20) + 1e-8)
        feats[f"{s}_rate_5v30"] = (r5 - r30) / (np.abs(r30) + 1e-8)
        feats[f"{s}_acc_5v20"] = r5 - r20
    return feats


def add_regime_residuals(df, sensors):
    """工况残差: 去除操作条件影响后的纯退化信号"""
    df = df.copy()
    df["s1_bin"] = pd.cut(df["setting_1"], bins=10, labels=False)
    df["s2_bin"] = pd.cut(df["setting_2"], bins=10, labels=False)
    df["regime"] = df["s1_bin"].astype(str) + "_" + df["s2_bin"].astype(str)

    feats = pd.DataFrame(index=df.index)
    means = df.groupby("regime")[sensors].transform("mean")
    for s in sensors:
        feats[f"{s}_resid"] = df[s] - means[s]
    return feats


def add_group_ratios(df, group_pos, group_neg):
    """传感器集团比值: 放大退化信号"""
    feats = pd.DataFrame(index=df.index)
    feat_pos = df[group_pos].mean(axis=1) + 1e-8
    feat_neg = df[group_neg].mean(axis=1) + 1e-8
    feats["group_pos_mean"] = feat_pos
    feats["group_neg_mean"] = feat_neg
    feats["ratio_pos_neg"] = feat_pos / feat_neg
    feats["log_ratio"] = np.log(feat_pos) - np.log(feat_neg)
    return feats


# ============================================================
# 主管线
# ============================================================
def build_pipeline(train_path, test_path, cfg):
    """完整特征工程管线: 加载 → 标签 → 特征 → 清洗"""
    sc = cfg["columns"]["sensors"]
    st = cfg["columns"]["settings"]
    wins = cfg["features"]["windows"]
    clip = cfg["features"]["rul_clip"]
    gp = cfg["features"]["group_pos"]
    gn = cfg["features"]["group_neg"]

    # --- 加载 ---
    train = load_raw(train_path)
    test = load_raw(test_path)

    # --- RUL 标签 ---
    train["RUL"] = compute_rul(train, clip=clip)
    # 测试集: 取每引擎最后一条记录, 用于评估
    last_idx = test.groupby("unit")["cycle"].idxmax()
    test_eval = test.loc[last_idx].reset_index(drop=True)

    # --- 特征工程 (训练集) ---
    def make_features(df):
        base = df[sc + st].copy()
        base.columns = [f"{c}_raw" if c in sc else c for c in base.columns]

        X = pd.concat(
            [
                base,
                add_rolling_features(df, sc, wins),
                add_trend_features(df, sc, wins),
                add_rate_features(df, sc),
                add_regime_residuals(df, sc),
                add_group_ratios(df, gp, gn),
            ],
            axis=1,
        )
        return X.bfill().ffill().fillna(0)

    X_train = make_features(train)
    X_test = make_features(test)
    X_test_eval = X_test.loc[last_idx].reset_index(drop=True)

    y_train = train["RUL"]

    print(f"[Pipeline] X_train: {X_train.shape}  |  y_train: {y_train.shape}")
    print(f"[Pipeline] X_test:  {X_test.shape}   |  X_test_eval: {X_test_eval.shape}")

    return X_train, y_train, X_test, X_test_eval, test_eval


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    cfg = load_config()
    X_train, y_train, X_test, X_test_eval, test_eval = build_pipeline(
        ROOT / cfg["data"]["train_path"],
        ROOT / cfg["data"]["test_path"],
        cfg,
    )
    # 缓存到 parquet
    X_train.to_parquet(ROOT / "cache_X_train.parquet")
    y_train.to_frame("RUL").to_parquet(ROOT / "cache_y_train.parquet")
    X_test_eval.to_parquet(ROOT / "cache_X_test_eval.parquet")
    print("[Pipeline] 特征已缓存到 cache_*.parquet")
