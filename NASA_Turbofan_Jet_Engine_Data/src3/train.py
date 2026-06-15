"""
NASA CMAPSS — 模型训练 & 评估
==============================
加载特征, 训练 XGBoost, 评估并保存模型。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
import yaml
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_pipeline import load_config, load_raw, build_pipeline

ROOT = Path(__file__).parent


# ============================================================
# 1. 数据准备
# ============================================================
def get_data(cfg):
    """加载或构建特征"""
    cache_x = ROOT / "cache_X_train.parquet"
    if cache_x.exists():
        print("[Train] 从缓存加载特征...")
        X = pd.read_parquet(cache_x)
        y = pd.read_parquet(ROOT / "cache_y_train.parquet")["RUL"]
        train_raw = load_raw(ROOT / cfg["data"]["train_path"])
    else:
        print("[Train] 运行特征管线...")
        X, y, _, _, _ = build_pipeline(
            ROOT / cfg["data"]["train_path"],
            ROOT / cfg["data"]["test_path"],
            cfg,
        )
        train_raw = load_raw(ROOT / cfg["data"]["train_path"])
    return X, y, train_raw


def split_by_engine(X, y, train_raw, test_size=0.2):
    """按发动机分组划分 train/val, 防止同发动机数据泄露"""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    idx_train, idx_val = next(gss.split(X, y, groups=train_raw["unit"]))
    return (
        X.iloc[idx_train],
        X.iloc[idx_val],
        y.iloc[idx_train],
        y.iloc[idx_val],
    )


# ============================================================
# 2. 训练
# ============================================================
def train_model(X_train, y_train, X_val, y_val, params, early_stopping=30):
    """训练 XGBoost, early stopping 基于验证集"""
    model = xgb.XGBRegressor(
        **params,
        early_stopping_rounds=early_stopping,
        eval_metric="rmse",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )
    return model


# ============================================================
# 3. 评估
# ============================================================
def evaluate(model, X_val, y_val, clip):
    """计算指标并绘图"""
    y_pred = model.predict(X_val)
    y_pred = np.clip(y_pred, 0, clip)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"\n[Eval] RMSE: {rmse:.2f}  |  MAE: {mae:.2f}  |  R2: {r2:.4f}")

    # ---- 图1: 预测 vs 真实 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(y_val, y_pred, alpha=0.5, s=20, color="steelblue")
    ax.plot([0, clip], [0, clip], "r--", lw=1.5, label="Perfect")
    ax.set_xlabel("True RUL"); ax.set_ylabel("Predicted RUL")
    ax.set_title(f"Predicted vs True (RMSE={rmse:.1f})")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    errors = y_pred - y_val
    ax.hist(errors, bins=20, color="steelblue", edgecolor="white")
    ax.axvline(0, color="red", ls="--")
    ax.axvline(errors.mean(), color="orange", ls="--", label=f"Bias={errors.mean():.1f}")
    ax.set_xlabel("Error"); ax.set_title(f"Error Distribution (std={errors.std():.1f})")
    ax.legend()

    plt.tight_layout(); plt.savefig(ROOT / "evaluation.png", dpi=150)
    plt.close(); print("[Eval] 图表已保存: evaluation.png")

    # ---- 图2: 学习曲线 ----
    evals = model.evals_result()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(evals["validation_0"]["rmse"], label="Train RMSE", alpha=0.7)
    ax.plot(evals["validation_1"]["rmse"], label="Val RMSE", alpha=0.7)
    ax.axvline(model.best_iteration, color="red", ls="--", label=f"Best={model.best_iteration}")
    ax.set_xlabel("Iteration"); ax.set_ylabel("RMSE")
    ax.set_title("Learning Curve"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(ROOT / "learning_curve.png", dpi=150)
    plt.close(); print("[Eval] 图表已保存: learning_curve.png")

    return rmse, mae


def feature_importance(model, X_train):
    """Top 20 特征重要性"""
    imp = (
        pd.DataFrame({"feature": X_train.columns, "gain": model.feature_importances_})
        .sort_values("gain", ascending=False)
        .head(20)
    )
    print("\n[Importance] Top 20:")
    for i, (_, r) in enumerate(imp.iterrows()):
        print(f"  {i+1:2d}. {r['feature']:<45s} {r['gain']:.4f}")

    # 柱状图
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(20), imp["gain"].values[::-1], color="steelblue")
    ax.set_yticks(range(20))
    ax.set_yticklabels(imp["feature"].values[::-1], fontsize=9)
    ax.set_xlabel("Gain"); ax.set_title("Top 20 Feature Importance")
    plt.tight_layout(); plt.savefig(ROOT / "feature_importance.png", dpi=150)
    plt.close(); print("[Importance] 图表已保存: feature_importance.png")


# ============================================================
# 主流程
# ============================================================
def main():
    cfg = load_config(ROOT / "config.yaml")
    clip = cfg["features"]["rul_clip"]

    # 数据
    X, y, train_raw = get_data(cfg)
    X_tr, X_val, y_tr, y_val = split_by_engine(
        X, y, train_raw, cfg["train"]["test_size"]
    )
    print(f"[Train] X_tr: {X_tr.shape}  |  X_val: {X_val.shape}")

    # 训练
    model = train_model(
        X_tr, y_tr, X_val, y_val,
        cfg["model"]["params"],
        cfg["train"]["early_stopping"],
    )
    print(f"[Train] 最佳迭代: {model.best_iteration}")

    # 评估
    evaluate(model, X_val, y_val, clip)
    feature_importance(model, X_tr)

    # 保存
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models/xgb_rul.pkl")
    print("[Train] 模型已保存: models/xgb_rul.pkl")


if __name__ == "__main__":
    main()
