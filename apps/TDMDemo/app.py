"""APPSolver multi-source ship-flow intelligent prediction demo."""

from __future__ import annotations

import html
import json
import shlex
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.TDMDemo.backend import (  # noqa: E402
    CHANNELS,
    MODEL_LABELS,
    OUTPUT_ROOT,
    TrainingConfig,
    available_checkpoints,
    build_training_command,
    checkpoint_dir,
    condition_summary,
    discover_conditions,
    load_rollout_metrics,
    load_training_metrics,
    read_job,
    result_to_csv,
    start_training,
    stop_training,
    tail_log,
)
from apps.TDMDemo.inference import APPPredictor, PredictionResult  # noqa: E402

st.set_page_config(
    page_title="TDM · 船舶流场智能预报",
    page_icon="≈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    f"<style>{(Path(__file__).with_name('style.css')).read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)

CHANNEL_LABELS = {
    "U:0": "纵向速度 u",
    "U:1": "横向速度 v",
    "U:2": "垂向速度 w",
    "p_rgh": "动压 p_rgh",
}
MODEL_DESCRIPTIONS = {
    "app_transformer": "以自适应四叉树划分承载几何先验，将非结构网格转化为 patch tokens。",
    "fno": "在规则潜空间执行谱卷积，以 Fourier modes 控制可解析的空间频率。",
    "pcno": "在点云上联合积分算子、梯度算子与邻域图，直接建模非结构网格。",
}
PLOT_COLORS = ["#08a7b5", "#f28b45", "#506b8b"]


def hero(title: str, copy: str, status: str = "SHIPBENCH · ONLINE") -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div><div class="hero-tag">APPSOLVER / TDM PLATFORM</div><h1>{title}</h1><p>{copy}</p></div>
          <div class="status-pill">● {status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f'<div class="section-kicker">{kicker}</div><h2 class="section-title">{title}</h2>'
        f'<p class="section-copy">{copy}</p>',
        unsafe_allow_html=True,
    )


def kpis(items: list[tuple[str, str, str]]) -> None:
    cards = "".join(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>'
        for label, value, note in items
    )
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)


def info_card(title: str, text: str, chip: str = "") -> None:
    chip_html = f'<span class="model-chip">{chip}</span>' if chip else ""
    st.markdown(
        f'<div class="info-card">{chip_html}<h4>{title}</h4><p>{text}</p></div>',
        unsafe_allow_html=True,
    )


def condition_keys() -> list[str]:
    return [
        f"{hull}/{reynolds}"
        for hull, conditions in discover_conditions().items()
        for reynolds in conditions
    ]


def get_predictor(condition: str, device: str) -> APPPredictor:
    key = (condition, device)
    if st.session_state.get("predictor_key") != key:
        st.session_state.predictor = APPPredictor(condition, device)
        st.session_state.predictor_key = key
    return st.session_state.predictor


def field_figure(result: PredictionResult, channel_index: int) -> go.Figure:
    coords = result.coords
    target = result.target[:, channel_index]
    prediction = result.prediction[:, channel_index]
    error = np.abs(prediction - target)
    if len(coords) > 14000:
        sample = np.linspace(0, len(coords) - 1, 14000, dtype=int)
        coords, target, prediction, error = (
            coords[sample], target[sample], prediction[sample], error[sample]
        )
    channel = CHANNELS[channel_index]
    low = float(min(target.min(), prediction.min()))
    high = float(max(target.max(), prediction.max()))
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("真实场", "预测场", "绝对误差"),
        horizontal_spacing=0.035,
    )
    for column, values, cmin, cmax, colorscale in (
        (1, target, low, high, "Viridis"),
        (2, prediction, low, high, "Viridis"),
        (3, error, 0.0, float(error.max()), "YlOrRd"),
    ):
        fig.add_trace(
            go.Scattergl(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker={
                    "size": 3,
                    "color": values,
                    "colorscale": colorscale,
                    "cmin": cmin,
                    "cmax": cmax,
                    "showscale": True,
                    "colorbar": {"thickness": 9, "len": 0.72, "x": column / 3 - 0.015},
                },
                hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>value=%{marker.color:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=column,
        )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_yaxes(scaleanchor="x3", scaleratio=1, row=1, col=3)
    fig.update_layout(
        title=f"{CHANNEL_LABELS[channel]} · t={result.target_step}",
        height=440,
        margin={"l": 15, "r": 15, "t": 70, "b": 20},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#30445d"},
    )
    return fig


def rollout_figure(frame: pd.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure()
    for metric, color, label in (
        ("rmse", "#08a7b5", "RMSE"),
        ("mae", "#506b8b", "MAE"),
        ("relative_l2", "#f28b45", "Relative L2"),
    ):
        fig.add_trace(
            go.Scatter(
                x=frame["horizon"], y=frame[metric], mode="lines+markers",
                name=label, line={"color": color, "width": 2}, marker={"size": 5},
            )
        )
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#d85a45",
        annotation_text=f"预警阈值 {threshold:.2f}", annotation_position="top left",
    )
    fig.update_layout(
        height=390,
        margin={"l": 20, "r": 20, "t": 35, "b": 20},
        xaxis_title="Rollout 步数",
        yaxis_title="归一化误差",
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={"orientation": "h", "y": 1.08},
    )
    return fig


def render_overview() -> None:
    catalog = discover_conditions()
    summaries = [condition_summary(key) for key in condition_keys()]
    total_points = sum(item["points"] for item in summaries)
    hero("多源数据–物理模型融合的智能预报组件", "从 ShipBench 数据接入、算子建模与融合训练，到单步预测、rollout 监测和异常预警。")
    kpis(
        [
            ("数据对象", f"{len(catalog)} 型", "DTC · KCS · KVLCC2"),
            ("流场工况", f"{len(summaries)} 组", "1Re / 2Re 多源融合"),
            ("时序帧", "350 / 工况", "固定参考网格"),
            ("网格点规模", f"{total_points / 1000:.0f}k", "六工况参考点合计"),
        ]
    )
    section("WORKFLOW", "端到端科研模型工作台", "所有模块围绕已有 APPSolver 与神经算子训练脚本构建，不引入 cfdBench。")
    cols = st.columns(4)
    steps = [
        ("01", "物理信息建模", "APP 几何划分、FNO 频域先验与 PCNO 点云邻域。"),
        ("02", "融合训练", "跨船型、跨雷诺数工况组合训练，统一配置与日志。"),
        ("03", "智能预报", "真实 APP 权重执行单步推理与自回归多步 rollout。"),
        ("04", "监测预警", "误差趋势、阈值事件、结果与配置统一导出。"),
    ]
    for col, (number, title, copy) in zip(cols, steps):
        with col:
            info_card(title, copy, number)

    st.markdown("<div class='flow-line'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("已训练基线 · DTC")
        rows = []
        for model in MODEL_LABELS:
            metrics = load_training_metrics("DTC", model)
            if not metrics.empty:
                last = metrics.iloc[-1]
                rows.append(
                    {
                        "模型": MODEL_LABELS[model],
                        "验证 MAE": last["mae"],
                        "验证 RMSE": last["rmse"],
                        "Relative L2": last["relative_l2"],
                    }
                )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with right:
        st.subheader("运行状态")
        checkpoints = available_checkpoints()
        job = read_job()
        info_card(
            "模型资产就绪",
            f"发现 {len(checkpoints)} 个 ShipBench checkpoint；训练任务状态："
            f"{job['status'] if job else 'idle'}。",
            "READY",
        )
        st.markdown(
            "<div class='notice'>实时推理使用本地 seed42 APP-Transformer 最优权重；"
            "单次前向延迟会在预测页按当前设备实测。</div>",
            unsafe_allow_html=True,
        )


def render_modeling() -> None:
    hero("物理信息建模", "以可调神经算子与 APP 几何超参数表达物理结构先验，而不是虚构项目中尚不存在的 PINN。", "MODEL LAB")
    section("PHYSICS-AWARE", "选择算子基线", "参数会同步用于训练命令生成；每种模型对应不同的空间物理归纳偏置。")
    model = st.segmented_control(
        "模型类型",
        options=list(MODEL_LABELS),
        format_func=lambda item: MODEL_LABELS[item],
        key="model_choice",
        default="app_transformer",
        label_visibility="collapsed",
    ) or "app_transformer"
    cols = st.columns(3)
    for col, key in zip(cols, MODEL_LABELS):
        with col:
            active = " active" if key == model else ""
            st.markdown(
                f'<div class="model-card{active}"><span class="model-chip">{MODEL_LABELS[key]}</span>'
                f'<h4>{"当前方案" if key == model else "候选基线"}</h4><p>{MODEL_DESCRIPTIONS[key]}</p></div>',
                unsafe_allow_html=True,
            )

    st.divider()
    left, right = st.columns([1, 1.05])
    with left:
        st.subheader("物理建模参数")
        if model == "app_transformer":
            st.number_input("Patch 容量", 64, 512, 256, 32, key="patch_size")
            st.slider("空间下采样比例", 0.2, 1.0, 0.6, 0.05, key="downsample_ratio")
            st.number_input("Token 维度 d_model", 32, 256, 56, 8, key="d_model")
            st.number_input("注意力头数", 1, 16, 4, 1, key="attention_heads")
            st.number_input("编码层数", 1, 10, 4, 1, key="model_layers")
        elif model == "fno":
            st.number_input("Fourier modes", 2, 20, 8, 1, key="fourier_modes")
            st.number_input("隐层宽度", 16, 128, 32, 8, key="hidden_width")
            st.number_input("谱卷积层数", 1, 10, 5, 1, key="model_layers")
        else:
            st.number_input("Fourier modes", 2, 16, 4, 1, key="fourier_modes")
            st.number_input("k-NN 邻居数", 4, 24, 8, 1, key="neighbors")
            st.number_input("算子宽度", 16, 128, 64, 8, key="hidden_width")
            st.number_input("W + K + D 层数", 1, 8, 4, 1, key="model_layers")
    with right:
        st.subheader("物理信息如何进入网络")
        explanations = {
            "app_transformer": [
                ("自适应空间划分", "船体近场采用更细 patch，显式保留自由液面局部结构。"),
                ("距离感知采样", "downsample ratio 决定每个 patch 内保留的物理观测密度。"),
                ("时序推进约束", "网络学习 u(t) → u(t+1)，并在 rollout 中递归推进。"),
            ],
            "fno": [
                ("频域截断", "Fourier modes 控制可学习波数范围，对应流场空间尺度。"),
                ("潜在规则网格", "非结构点经几何投影进入谱域，再回到原始点集。"),
                ("共享算子", "跨网格位置共享频域核，形成分辨率无关的基线。"),
            ],
            "pcno": [
                ("积分算子 K", "基于点云 quadrature 的 Fourier 积分传播全局信息。"),
                ("微分算子 D", "k-NN 最小二乘梯度编码局部变化率。"),
                ("点算子 W", "局部线性映射与 K、D 在每层共同更新流场。"),
            ],
        }
        for title, text in explanations[model]:
            info_card(title, text)
            st.write("")
        st.markdown(
            "<div class='notice'>本页将“物理信息网络建模”落实为已有代码可执行的物理归纳偏置与超参数配置；不声称实现 PINN 方程残差。</div>",
            unsafe_allow_html=True,
        )


def current_training_config() -> TrainingConfig:
    model = st.session_state.get("model_choice", "app_transformer")
    selected = tuple(st.session_state.get("train_conditions", condition_keys()[:2]))
    return TrainingConfig(
        model=model,
        conditions=selected,
        batch_size=int(st.session_state.get("batch_size", 2)),
        max_steps=int(st.session_state.get("max_steps", 1000)),
        eval_every=int(st.session_state.get("eval_every", 200)),
        learning_rate=float(st.session_state.get("learning_rate", 1e-4)),
        train_ratio=float(st.session_state.get("train_ratio", 0.8)),
        rollout_holdout_steps=int(st.session_state.get("holdout", 50)),
        seed=int(st.session_state.get("seed", 42)),
        patch_size=int(st.session_state.get("patch_size", 256)),
        downsample_ratio=float(st.session_state.get("downsample_ratio", 0.6)),
        d_model=int(st.session_state.get("d_model", 56)),
        attention_heads=int(st.session_state.get("attention_heads", 4)),
        layers=int(st.session_state.get("model_layers", 4)),
        hidden_width=int(st.session_state.get("hidden_width", 32)),
        fourier_modes=int(st.session_state.get("fourier_modes", 8)),
        neighbors=int(st.session_state.get("neighbors", 8)),
    )


def render_training() -> None:
    hero("物理–数据驱动融合训练", "选择多个 ShipBench 工况，生成并启动项目原生训练任务；页面保留参数、日志与模型产物。", "TRAINING CONSOLE")
    left, right = st.columns([1, 1])
    with left:
        section("DATA FUSION", "训练数据与模型", "多选工况会启用项目的 multi-condition 训练模式。")
        st.selectbox(
            "模型",
            options=list(MODEL_LABELS),
            format_func=lambda item: MODEL_LABELS[item],
            key="model_choice",
        )
        st.multiselect(
            "ShipBench 工况",
            options=condition_keys(),
            default=condition_keys()[:2],
            key="train_conditions",
        )
        c1, c2 = st.columns(2)
        c1.number_input("Batch size", 1, 16, 2, 1, key="batch_size")
        c2.number_input("最大训练步数", 10, 100000, 1000, 100, key="max_steps")
        c1.number_input("验证间隔", 10, 10000, 200, 10, key="eval_every")
        c2.number_input("随机种子", 0, 10000, 42, 1, key="seed")
        c1.number_input("学习率", 1e-6, 1e-2, 1e-4, format="%.1e", key="learning_rate")
        c2.slider("训练集比例", 0.6, 0.9, 0.8, 0.05, key="train_ratio")
        st.number_input("Rollout 保留窗口", 5, 100, 50, 5, key="holdout")
        config = current_training_config()
        st.caption("模型专用参数取自“物理信息建模”页面；可在启动前返回调整。")

    with right:
        section("EXECUTION", "运行计划", "训练以独立本地进程运行，产物写入 outputs/tdm_demo（已被 git 忽略）。")
        if config.conditions:
            preview_dir = OUTPUT_ROOT / "<timestamp_model>"
            command = build_training_command(config, preview_dir)
            st.code(shlex.join(command), language="bash", wrap_lines=True)
        else:
            command = []
            st.warning("至少选择一个 ShipBench 工况。")
        b1, b2, b3 = st.columns([1, 1, 1.3])
        if b1.button("启动训练", type="primary", disabled=not command, width="stretch"):
            try:
                start_training(config)
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))
        if b2.button("停止任务", width="stretch"):
            stop_training()
            st.rerun()
        if b3.download_button(
            "导出训练配置",
            data=json.dumps(asdict(config), ensure_ascii=False, indent=2),
            file_name="tdm_training_config.json",
            mime="application/json",
            width="stretch",
        ):
            pass

        job = read_job()
        if job:
            status_text = "运行中" if job["status"] == "running" else "已结束"
            kpis(
                [
                    ("任务状态", status_text, f"PID {job['pid']}"),
                    ("启动时间", job["started_at"].split("T")[-1], Path(job["run_dir"]).name),
                    ("运行设备", "AUTO", "CUDA / MPS / CPU"),
                    ("输出目录", "已创建", "outputs/tdm_demo"),
                ]
            )
            log = tail_log(job["log_path"])
        else:
            log = "$ 等待训练任务启动…"
        st.markdown(
            f'<div class="terminal">{html.escape(log)}</div>', unsafe_allow_html=True
        )
        if job and job["status"] == "running" and st.button("刷新终端"):
            st.rerun()


def render_prediction_result(result: PredictionResult, threshold: float) -> None:
    metrics = result.metrics
    status = "正常" if metrics["rmse"] <= threshold else "预警"
    kpis(
        [
            ("RMSE", f"{metrics['rmse']:.4f}", "归一化全通道"),
            ("MAE", f"{metrics['mae']:.4f}", "归一化全通道"),
            ("Relative L2", f"{metrics['relative_l2']:.4f}", "全场相对误差"),
            ("推理耗时", f"{result.latency_seconds * 1000:.1f} ms", f"状态：{status}"),
        ]
    )
    if result.latency_seconds >= 10:
        st.error("本次智能预报计算超过 10 秒。")
    channel = st.selectbox(
        "可视化变量", range(4), format_func=lambda index: CHANNEL_LABELS[CHANNELS[index]]
    )
    st.plotly_chart(field_figure(result, channel), width="stretch", config={"displaylogo": False})

    e1, e2, e3 = st.columns(3)
    e1.download_button(
        "导出预测场 CSV",
        result_to_csv(result.coords, result.prediction, result.target),
        file_name=f"prediction_t{result.target_step}.csv",
        mime="text/csv",
        width="stretch",
    )
    e2.download_button(
        "导出指标 JSON",
        json.dumps(result.metrics, ensure_ascii=False, indent=2),
        file_name=f"metrics_t{result.target_step}.json",
        mime="application/json",
        width="stretch",
    )
    checkpoint = checkpoint_dir(st.session_state.predict_condition.split("/")[0], "app_transformer") / "model_best_mae.pth"
    if checkpoint.is_file():
        e3.download_button(
            "导出模型权重",
            checkpoint.read_bytes(),
            file_name=checkpoint.name,
            mime="application/octet-stream",
            width="stretch",
        )


def render_prediction() -> None:
    hero("智能预报与异常趋势预警", "APP-Transformer 真实权重执行单步预测或自回归 rollout；阈值由归一化 RMSE 触发。", "INFERENCE READY")
    controls, workspace = st.columns([0.78, 2.05])
    with controls:
        section("INPUT", "预测任务", "输入来自 ShipBench 固定参考网格。")
        condition = st.selectbox("船型 / 工况", condition_keys(), key="predict_condition")
        summary = condition_summary(condition)
        source_step = st.slider(
            "起始时间步", 0, max(0, summary["frames"] - 2), min(300, summary["frames"] - 2)
        )
        mode = st.radio("预测模式", ["单步预测", "多步 Rollout"], horizontal=True)
        horizon = st.slider("Rollout 步数", 2, 30, 10, disabled=mode == "单步预测")
        threshold = st.slider("异常预警阈值（RMSE）", 0.05, 1.0, 0.2, 0.01)
        device = st.selectbox("计算设备", ["auto", "mps", "cpu"], index=0)
        st.markdown(
            f"<div class='notice'>数据：{summary['frames']} 帧 · {summary['points']:,} 点<br>"
            "模型：APP-Transformer · seed42 best MAE</div>",
            unsafe_allow_html=True,
        )
        run = st.button("执行智能预报", type="primary", width="stretch")

    if run:
        with st.spinner("正在加载网格、权重并计算…"):
            predictor = get_predictor(condition, device)
            if mode == "单步预测":
                st.session_state.single_result = predictor.predict_step(source_step)
                st.session_state.rollout_results = None
            else:
                allowed_horizon = min(horizon, predictor.frame_count - source_step - 1)
                results, total = predictor.rollout(source_step, allowed_horizon)
                st.session_state.rollout_results = (results, total, threshold)
                st.session_state.single_result = results[0]

    with workspace:
        result = st.session_state.get("single_result")
        rollout_data = st.session_state.get("rollout_results")
        if result is None:
            section("OUTPUT", "等待计算", "配置左侧任务后开始单步或多步预报。")
            kpis(
                [
                    ("模型状态", "READY", "本地权重可用"),
                    ("输入通道", "4", "u · v · w · p_rgh"),
                    ("预测步长", "Δt = 1", "自回归时序推进"),
                    ("时延目标", "< 10 s", "单步智能预报"),
                ]
            )
            st.image(PROJECT_ROOT / "pics" / "qualitative_app_transformer_uvwp.png", width="stretch")
        else:
            render_prediction_result(result, threshold)
            if rollout_data:
                results, total, used_threshold = rollout_data
                frame = pd.DataFrame(
                    [
                        {
                            "horizon": index,
                            "mae": item.metrics["mae"],
                            "rmse": item.metrics["rmse"],
                            "relative_l2": item.metrics["relative_l2"],
                            "latency_ms": item.latency_seconds * 1000,
                        }
                        for index, item in enumerate(results, start=1)
                    ]
                )
                st.subheader("Rollout 误差趋势")
                st.plotly_chart(rollout_figure(frame, used_threshold), width="stretch")
                alerts = frame[frame["rmse"] > used_threshold]
                if alerts.empty:
                    st.success(f"{len(frame)} 步 rollout 均未超过阈值 {used_threshold:.2f}。")
                else:
                    first = int(alerts.iloc[0]["horizon"])
                    st.error(
                        f"异常趋势预警：第 {first} 步首次超过阈值 {used_threshold:.2f}，"
                        f"共 {len(alerts)} 个预警步。"
                    )
                d1, d2 = st.columns(2)
                d1.metric("Rollout 总计算耗时", f"{total:.3f} s")
                d2.download_button(
                    "导出 Rollout 指标",
                    frame.to_csv(index=False).encode("utf-8"),
                    file_name="rollout_metrics.csv",
                    mime="text/csv",
                    width="stretch",
                )


def render_monitoring() -> None:
    hero("预报监测仪表盘", "回放已完成的 ShipBench 多步评估，比较 APP、FNO 与 PCNO，并按统一阈值产生趋势预警。", "MONITORING")
    hull = st.selectbox("船型", list(discover_conditions()))
    threshold = st.slider("RMSE 预警阈值", 0.05, 1.0, 0.3, 0.01, key="monitor_threshold")
    frames = {}
    figure = go.Figure()
    for model, color in zip(MODEL_LABELS, PLOT_COLORS):
        frame = load_rollout_metrics(hull, model)
        if frame.empty:
            continue
        frames[model] = frame
        figure.add_trace(
            go.Scatter(
                x=frame["horizon"], y=frame["rmse"], mode="lines",
                name=MODEL_LABELS[model], line={"width": 2.4, "color": color},
            )
        )
    figure.add_hline(
        y=threshold, line_dash="dash", line_color="#d85a45",
        annotation_text=f"阈值 {threshold:.2f}",
    )
    figure.update_layout(
        height=440,
        xaxis_title="Rollout horizon",
        yaxis_title="Normalized RMSE",
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={"orientation": "h", "y": 1.08},
        margin={"l": 25, "r": 20, "t": 35, "b": 25},
    )
    st.plotly_chart(figure, width="stretch")

    cards = st.columns(3)
    for col, model in zip(cards, MODEL_LABELS):
        with col:
            frame = frames.get(model)
            if frame is None:
                continue
            exceeded = frame[frame["rmse"] > threshold]
            first = "未触发" if exceeded.empty else f"H={int(exceeded.iloc[0]['horizon'])}"
            info_card(
                MODEL_LABELS[model],
                f"末步 RMSE {frame.iloc[-1]['rmse']:.3f} · 首次预警 {first}",
                "NORMAL" if exceeded.empty else "ALERT",
            )
    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("训练与终端")
        job = read_job()
        if job:
            st.markdown(
                f'<div class="terminal">{html.escape(tail_log(job["log_path"], 35))}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="terminal">$ 当前无训练任务</div>', unsafe_allow_html=True)
    with right:
        st.subheader("导出中心")
        if frames:
            combined = pd.concat(
                [frame.assign(model=MODEL_LABELS[model]) for model, frame in frames.items()],
                ignore_index=True,
            )
            st.download_button(
                "导出模型对比 CSV",
                combined.to_csv(index=False).encode("utf-8"),
                file_name=f"{hull}_rollout_comparison.csv",
                mime="text/csv",
                width="stretch",
            )
        manifest = {
            "dataset": "shipBench",
            "hull": hull,
            "models": list(MODEL_LABELS.values()),
            "alert_metric": "normalized_rmse",
            "threshold": threshold,
        }
        st.download_button(
            "导出监测配置 JSON",
            json.dumps(manifest, ensure_ascii=False, indent=2),
            file_name=f"{hull}_monitor_config.json",
            mime="application/json",
            width="stretch",
        )
        st.markdown(
            "<div class='notice'>导出内容包含预测结果、训练配置、评估指标与本地模型权重；数据集和运行产物不进入 git。</div>",
            unsafe_allow_html=True,
        )


with st.sidebar:
    st.markdown("## ≈ TDM")
    st.caption("SHIP FLOW INTELLIGENCE")
    st.markdown("---")
    page = st.radio(
        "工作台",
        ["总览", "物理信息建模", "融合训练", "智能预测", "监测与导出"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("数据范围")
    st.markdown("**ShipBench only**")
    st.caption("运行环境")
    st.code("/opt/miniconda3/envs/mesh", language=None)
    job = read_job()
    st.caption(f"训练任务 · {job['status'] if job else 'idle'}")

if page == "总览":
    render_overview()
elif page == "物理信息建模":
    render_modeling()
elif page == "融合训练":
    render_training()
elif page == "智能预测":
    render_prediction()
else:
    render_monitoring()
