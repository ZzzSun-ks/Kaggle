"""
NASA CMAPSS — 在线推理引擎
============================
加载训练好的模型, 对新数据进行 RUL 预测与健康状态判定。
"""
import numpy as np
import pandas as pd
import joblib
import yaml
from pathlib import Path
from data_pipeline import load_config, load_raw, compute_rul

ROOT = Path(__file__).parent


class InferenceEngine:
    """在线推理: 加载模型 → 预处理 → 预测 RUL → 健康判定"""

    def __init__(self, model_path=None, config_path=None):
        cfg = load_config(config_path or ROOT / "config.yaml")
        self.clip = cfg["features"]["rul_clip"]
        self.thresh_warn = cfg["inference"]["threshold_warning"]
        self.thresh_crit = cfg["inference"]["threshold_critical"]
        self.columns = cfg["columns"]

        model_path = model_path or ROOT / cfg["inference"]["model_path"]
        self.model = joblib.load(model_path)
        print(f"[Inference] 模型已加载: {model_path}")

    def predict(self, data, feature_pipeline=None):
        """
        对特征矩阵直接预测 RUL。

        Args:
            data: 特征 DataFrame 或原始 DataFrame (需提供 pipeline)
            feature_pipeline: 特征工程函数, 签名为 fn(df) -> X
        Returns:
            np.ndarray: RUL 预测值 (clipped)
        """
        if feature_pipeline is not None:
            X = feature_pipeline(data)
        else:
            X = data
        rul = self.model.predict(X)
        return np.clip(rul, 0, self.clip)

    def assess_health(self, rul_pred):
        """根据预测 RUL 判定健康状态

        Returns:
            list[dict]: 每个样本的 {rul, status, action}
        """
        results = []
        for r in np.atleast_1d(rul_pred):
            if r <= self.thresh_crit:
                status, action = "CRITICAL", "立即停机维修"
            elif r <= self.thresh_warn:
                status, action = "WARNING", "安排维护计划"
            else:
                status, action = "NORMAL", "继续运行"
            results.append({"rul": round(float(r), 1), "status": status, "action": action})
        return results

    def predict_engine(self, engine_data, feature_pipeline):
        """
        对单个发动机的全生命周期数据预测 RUL 退化曲线。

        Returns:
            DataFrame: cycle, rul_pred, status
        """
        X = feature_pipeline(engine_data)
        rul = self.predict(X)
        health = self.assess_health(rul)
        return pd.DataFrame(
            {
                "cycle": engine_data["cycle"].values,
                "rul_pred": [h["rul"] for h in health],
                "status": [h["status"] for h in health],
            }
        )


# ============================================================
# CLI: 演示
# ============================================================
if __name__ == "__main__":
    cfg = load_config(ROOT / "config.yaml")

    # 加载测试集并做预测
    from data_pipeline import build_pipeline

    _, _, X_test, X_test_eval, test_eval = build_pipeline(
        ROOT / cfg["data"]["train_path"],
        ROOT / cfg["data"]["test_path"],
        cfg,
    )

    engine = InferenceEngine()
    rul_pred = engine.predict(X_test_eval)
    health = engine.assess_health(rul_pred)

    # 输出前 20 个发动机的预测结果
    print("\n[Inference] 测试集预测 (前20台发动机):")
    print(f"{'Engine':<8} {'RUL_pred':>10} {'Status':<12} {'Action'}")
    print("-" * 60)
    for i, h in enumerate(health[:20]):
        print(f"#{i+1:<7} {h['rul']:>10.1f} {h['status']:<12} {h['action']}")

    # 统计
    warns = sum(1 for h in health if h["status"] == "WARNING")
    crits = sum(1 for h in health if h["status"] == "CRITICAL")
    norms = len(health) - warns - crits
    print(f"\n  正常: {norms}  |  警告: {warns}  |  严重: {crits}")
