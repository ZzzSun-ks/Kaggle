"""
NASA CMAPSS — Streamlit 预测性维护看板
========================================
可视化: 数据概览, 模型性能, 单引擎退化曲线, 批量预测。
运行: streamlit run dashboard.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from data_pipeline import load_config, load_raw, build_pipeline
from inference import InferenceEngine

ROOT = Path(__file__).parent
cfg = load_config(ROOT / "config.yaml")

st.set_page_config(page_title="预测性维护看板", layout="wide")
st.title("🛩️ NASA CMAPSS — 预测性维护看板")
st.markdown("涡轮风扇发动机剩余使用寿命 (RUL) 预测")

# ---- 侧边栏 ----
st.sidebar.header("⚙️ 配置")
data_mode = st.sidebar.radio("数据模式", ["训练集", "测试集"])
engine_id = st.sidebar.number_input("发动机编号", min_value=1, max_value=100, value=1)

# ---- 加载数据 ----
@st.cache_data
def load_data():
    train = load_raw(ROOT / cfg["data"]["train_path"])
    test = load_raw(ROOT / cfg["data"]["test_path"])
    return train, test

train, test = load_data()

# ---- 加载模型 ----
@st.cache_resource
def load_model():
    path = ROOT / cfg["inference"]["model_path"]
    if path.exists():
        return InferenceEngine(str(path))
    return None

engine = load_model()

# ============================================================
# Tab 1: 数据概览
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 数据概览", "📈 模型评估", "🔍 单引擎分析", "📋 批量预测"])

with tab1:
    st.header("数据集概览")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("训练集")
        st.metric("发动机数量", train["unit"].nunique())
        st.metric("总记录数", len(train))
        st.metric("传感器数", len(cfg["columns"]["sensors"]))
        st.dataframe(train.describe().round(3), use_container_width=True)

    with col2:
        st.subheader("测试集")
        st.metric("发动机数量", test["unit"].nunique())
        st.metric("总记录数", len(test))
        max_cycle = test.groupby("unit")["cycle"].max()
        st.metric("每引擎平均 cycles", f"{max_cycle.mean():.0f}")
        st.dataframe(test.describe().round(3), use_container_width=True)

    st.subheader("传感器时序示例 (Engine #1, sensor s2)")
    fig, ax = plt.subplots(figsize=(10, 3))
    eng1 = train[train["unit"] == 1]
    ax.plot(eng1["cycle"], eng1["s2"], color="steelblue", lw=1)
    ax.set_xlabel("Cycle"); ax.set_ylabel("s2")
    ax.set_title("Engine #1 — Sensor s2 退化曲线")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# ============================================================
# Tab 2: 模型评估
# ============================================================
with tab2:
    st.header("模型性能")

    eval_path = ROOT / "evaluation.png"
    lc_path = ROOT / "learning_curve.png"
    fi_path = ROOT / "feature_importance.png"

    if eval_path.exists() and lc_path.exists():
        col1, col2 = st.columns(2)
        with col1:
            st.image(str(eval_path), caption="预测 vs 真实 & 误差分布")
        with col2:
            st.image(str(lc_path), caption="学习曲线")
    else:
        st.warning("请先运行 train.py 生成评估图表")

    if fi_path.exists():
        st.image(str(fi_path), caption="特征重要性 Top 20")
    else:
        st.info("运行 train.py 生成特征重要性图")

    if engine is not None:
        st.subheader("模型信息")
        st.json(
            {
                "model_type": type(engine.model).__name__,
                "rul_clip": engine.clip,
                "warning_threshold": engine.thresh_warn,
                "critical_threshold": engine.thresh_crit,
            }
        )

# ============================================================
# Tab 3: 单引擎分析
# ============================================================
with tab3:
    st.header(f"发动机 #{engine_id} 退化分析")

    source = train if data_mode == "训练集" else test
    eng = source[source["unit"] == engine_id].copy()
    cycles = eng["cycle"].values

    if len(eng) == 0:
        st.error(f"发动机 #{engine_id} 不存在")
    else:
        # 传感器选择
        sensor = st.selectbox("选择传感器", cfg["columns"]["sensors"])

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # 左: 单传感器退化曲线
        ax = axes[0]
        ax.plot(cycles, eng[sensor], "o-", color="steelblue", ms=3, lw=1)
        ax.set_xlabel("Cycle"); ax.set_ylabel(sensor)
        ax.set_title(f"Engine #{engine_id} — {sensor}")
        ax.grid(alpha=0.3)

        # 右: 多传感器热力图
        ax = axes[1]
        sensor_cols = cfg["columns"]["sensors"]
        sensor_data = eng[sensor_cols].values.T
        # 归一化到 [0,1]
        vmin, vmax = sensor_data.min(), sensor_data.max()
        im = ax.imshow(
            sensor_data, aspect="auto", cmap="RdYlGn_r",
            vmin=vmin, vmax=vmax, interpolation="bilinear",
        )
        ax.set_xlabel("Cycle"); ax.set_ylabel("Sensor")
        ax.set_yticks(range(len(sensor_cols)))
        ax.set_yticklabels(sensor_cols, fontsize=7)
        ax.set_title(f"Engine #{engine_id} — 传感器热力图")
        plt.colorbar(im, ax=ax)

        st.pyplot(fig)

        # RUL 预测曲线
        if engine is not None:
            st.subheader("RUL 预测退化曲线")
            try:
                result = engine.predict_engine(eng, feature_pipeline=None)
                # 直接用原始特征预测需要 pipeline, 这里展示数据结构
                st.info("💡 如需完整的退化预测曲线, 请连接特征管线后运行。当前展示测试集评估结果。")
            except Exception as e:
                st.warning(f"模型预测需要特征管线: {e}")

# ============================================================
# Tab 4: 批量预测
# ============================================================
with tab4:
    st.header("批量健康评估")

    if engine is None:
        st.error("模型未找到, 请先运行 train.py 训练模型")
    else:
        if st.button("🔄 对测试集 100 台发动机运行评估"):
            with st.spinner("推理中..."):
                # 加载缓存的特征
                cache_path = ROOT / "cache_X_test_eval.parquet"
                if cache_path.exists():
                    X_test = pd.read_parquet(cache_path)
                else:
                    _, _, _, X_test, _ = build_pipeline(
                        ROOT / cfg["data"]["train_path"],
                        ROOT / cfg["data"]["test_path"],
                        cfg,
                    )

                rul_pred = engine.predict(X_test)
                health = engine.assess_health(rul_pred)

                df = pd.DataFrame(
                    {
                        "Engine": [f"#{i+1}" for i in range(len(health))],
                        "RUL_pred": [h["rul"] for h in health],
                        "Status": [h["status"] for h in health],
                        "Action": [h["action"] for h in health],
                    }
                )

                # 统计卡片
                warns = sum(1 for h in health if "警告" in h["status"])
                crits = sum(1 for h in health if "严重" in h["status"])
                norms = len(health) - warns - crits

                cols = st.columns(4)
                cols[0].metric("总发动机", len(health))
                cols[1].metric("🟢 正常", norms)
                cols[2].metric("🟡 警告", warns)
                cols[3].metric("🔴 严重", crits)

                st.dataframe(df, use_container_width=True, height=400)

                # RUL 分布图
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(range(len(rul_pred)), sorted(rul_pred, reverse=True), color="steelblue")
                ax.axhline(engine.thresh_warn, color="orange", ls="--", label=f"Warn={engine.thresh_warn}")
                ax.axhline(engine.thresh_crit, color="red", ls="--", label=f"Crit={engine.thresh_crit}")
                ax.set_xlabel("Engine Rank"); ax.set_ylabel("Predicted RUL")
                ax.set_title("测试集 RUL 预测分布 (降序)")
                ax.legend(); ax.grid(alpha=0.3)
                st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.caption("NASA CMAPSS FD001 | Predictive Maintenance")
