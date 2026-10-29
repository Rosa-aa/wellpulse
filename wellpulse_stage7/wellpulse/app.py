"""WellPulse: downhole-pressure soft sensor dashboard (Volve daily production data)."""
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="WellPulse", page_icon="🛢️", layout="wide")

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "hgb_v2.joblib"
DATA_PATH = BASE / "models" / "app_data.csv"
METRICS_PATH = BASE / "models" / "metrics.json"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


if not (MODEL_PATH.exists() and DATA_PATH.exists() and METRICS_PATH.exists()):
    st.error("Model files not found. Put Volve_production_data.xlsx into data/ and run: "
             "`python -m src.train`")
    st.stop()

art, data = load_model(), load_data()
report = json.loads(METRICS_PATH.read_text())
features = art["features"]

st.title("WellPulse")
st.caption("Estimating downhole pressure from surface and operational data "
           "(soft sensor, Equinor Volve field, 5 production wells)")

tab_wells, tab_predict, tab_metrics, tab_limits = st.tabs(
    ["Well explorer", "What-if prediction", "Metrics", "Limitations"])

# ---------------------------------------------------------------- Well explorer
with tab_wells:
    wells = sorted(data["well"].unique())
    well = st.selectbox("Well", wells)
    d = data[data["well"] == well].sort_values("date")
    show_flags = st.checkbox("Highlight flagged days", value=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days in dataset", f"{len(d):,}")
    c2.metric("MAE, bar (out-of-fold)", f"{d['residual'].abs().mean():.1f}")
    c3.metric("RMSE, bar (out-of-fold)", f"{(d['residual'] ** 2).mean() ** 0.5:.1f}")
    c4.metric("Flagged days", f"{int(d['is_anomaly'].sum())} ({d['is_anomaly'].mean():.1%})")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["date"], y=d[TARGET], mode="lines", name="Measured", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=d["date"], y=d["pred_oof"], mode="lines", name="Predicted (out-of-fold)",
                             line=dict(width=1.5, dash="dot")))
    if show_flags:
        f = d[d["is_anomaly"]]
        fig.add_trace(go.Scatter(x=f["date"], y=f[TARGET], mode="markers", name="Flagged (|residual| > 3σ)",
                                 marker=dict(size=7, symbol="x", color="red")))
    fig.update_layout(yaxis_title="Downhole pressure, bar", height=430, margin=dict(t=30, b=10),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

    fig_r = go.Figure(go.Scatter(x=d["date"], y=d["residual"], mode="markers", marker=dict(size=4)))
    thr = 3 * report["anomaly_sigma_bar"]
    med = report["anomaly_median_residual"]
    for y in (med + thr, med - thr):
        fig_r.add_hline(y=y, line_dash="dash", line_color="red")
    fig_r.update_layout(title="Residual (measured − predicted), bar", height=280, margin=dict(t=40, b=10))
    st.plotly_chart(fig_r, width="stretch")

    st.caption("A flag means the model cannot explain that day well. It is a prompt to investigate "
               "(sensor error, transient, real event or model weakness), not a fault diagnosis.")
    with st.expander("Flagged days table"):
        st.dataframe(d.loc[d["is_anomaly"], ["date", TARGET, "pred_oof", "residual"]]
                     .round({TARGET: 1, "pred_oof": 1, "residual": 1}),
                     width="stretch", hide_index=True)

# ---------------------------------------------------------------- What-if
with tab_predict:
    st.write("Set the operating conditions and get a downhole pressure estimate from the final model. "
             "Sliders are limited to ranges seen in the training data; the model does not extrapolate reliably.")
    labels = {
        "AVG_WHP_P": "Wellhead pressure, bar", "AVG_WHT_P": "Wellhead temperature, °C",
        "AVG_CHOKE_SIZE_P": "Choke size, %", "ON_STREAM_HRS": "Hours on stream",
        "BORE_OIL_VOL": "Oil volume (allocated)", "BORE_GAS_VOL": "Gas volume (allocated)",
        "BORE_WAT_VOL": "Water volume (allocated)", "days_on_prod": "Days since first production",
        "cum_oil": "Cumulative oil (allocated)", "cum_water": "Cumulative water (allocated)",
    }
    ref_well = st.selectbox("Start from the median conditions of well", sorted(data["well"].unique()), key="ref")
    base = data[data["well"] == ref_well][features].median()
    cols = st.columns(3)
    values = {}
    inputs = [f for f in features if f != "water_cut"]
    for i, f in enumerate(inputs):
        lo, hi = float(data[f].min()), float(data[f].max())
        values[f] = cols[i % 3].slider(labels.get(f, f), lo, hi, float(min(max(base[f], lo), hi)),
                                       key=f"{ref_well}_{f}")
    liquid = values["BORE_OIL_VOL"] + values["BORE_WAT_VOL"]
    values["water_cut"] = values["BORE_WAT_VOL"] / liquid if liquid > 0 else 0.0
    st.write(f"Computed water cut: **{values['water_cut']:.1%}**")
    row = pd.DataFrame([values])[features]
    pred = float(art["model"].predict(row)[0])
    st.metric("Estimated downhole pressure", f"{pred:.1f} bar",
              help=f"Typical error on a known well: about ±{report['time_split']['RMSE']:.0f} bar (RMSE).")
    st.caption(f"Typical error: RMSE {report['time_split']['RMSE']} bar on later days of known wells, "
               f"{report['lowo_mean_rmse']} bar on a well left out of training.")

# ---------------------------------------------------------------- Metrics
with tab_metrics:
    ts, cv = report["time_split"], report["blocked_cv_oof"]
    st.subheader("Evaluation (HistGradientBoosting, V2 features)")
    table = pd.DataFrame({
        "Setting": ["Per-well time split (first 80% train, last 20% test)",
                    "Leave-one-well-out, mean RMSE",
                    "Blocked 5-fold out-of-fold (dashboard plots only)"],
        "MAE": [ts["MAE"], None, cv["MAE"]],
        "RMSE": [ts["RMSE"], report["lowo_mean_rmse"], cv["RMSE"]],
        "R²": [ts["R2"], None, cv["R2"]],
    })
    st.dataframe(table, width="stretch", hide_index=True)
    st.markdown(
        "- Official numbers are the **time split** and **leave-one-well-out** rows.\n"
        "- Out-of-fold blocks are complementary, so that row is not a true forecast; it only feeds the plots.\n"
        "- Baselines on the same time split: global mean RMSE 28.6, per-well mean 21.4 bar.\n"
        "- Single split, untuned models, 5 wells.")

# ---------------------------------------------------------------- Limitations
with tab_limits:
    st.markdown(
        """
**What this is.** A soft sensor trained on the public Equinor Volve dataset: it estimates downhole pressure
from surface and operational variables. It is an independent preparation project, not a hackathon entry.

**Limitations**
- Only 5 production wells. Generalisation to a new well is partial (mean RMSE 16.5 bar) and uneven.
- Volumes (oil, gas, water) and the history features built from them are *allocated*, not measured.
- History features need the well's full production history from day one.
- Pressure drifts over the years, so predictions for later periods are harder.
- `AVG_DP_TUBING` and `AVG_DOWNHOLE_TEMPERATURE` are excluded on purpose (leakage / same downhole gauge).
- Anomaly flags are residual-based, uncalibrated and unlabeled: no precision/recall can be reported.
  A flag is not a fault diagnosis. F-12 H has a high flag rate, likely from model weakness on a short record.

**Data.** Equinor Volve Data Village, Equinor Open Data Licence. Raw data is not included in this repository.
        """
    )
