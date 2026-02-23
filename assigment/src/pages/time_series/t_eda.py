import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from pages.time_series.t_data_utils import df_weekly_ts

series = df_weekly_ts.set_index("Date")["Revenue"]


# helpers
def make_acf_chart(s, nlags, alpha=0.05, title="ACF"):
    n = len(s)
    vals, ci = acf(s.dropna(), nlags=nlags, alpha=alpha, fft=True)
    crit = 1.96 / np.sqrt(n)
    df = pd.DataFrame({"Lag": np.arange(len(vals)), "Value": vals})

    bars = (
        alt.Chart(df)
        .mark_bar(size=6)
        .encode(
            x=alt.X("Lag:Q", title="Lag"),
            y=alt.Y("Value:Q", title="ACF", scale=alt.Scale(domain=[-1, 1])),
            color=alt.condition(alt.datum.Value > 0, alt.value("#4C78A8"), alt.value("#E45756")),
            tooltip=["Lag:Q", alt.Tooltip("Value:Q", format=".3f")],
        )
    )
    ci_df = pd.DataFrame({"y": [crit, -crit]})
    ci_lines = (
        alt.Chart(ci_df)
        .mark_rule(strokeDash=[4, 4], color="grey", opacity=0.8)
        .encode(y="y:Q")
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="black").encode(y="y:Q")
    return (bars + ci_lines + zero).properties(title=title, width="container", height=280).interactive()


def make_pacf_chart(s, nlags, alpha=0.05, title="PACF"):
    n = len(s)
    vals, ci = pacf(s.dropna(), nlags=nlags, alpha=alpha, method="ywm")
    crit = 1.96 / np.sqrt(n)
    df = pd.DataFrame({"Lag": np.arange(len(vals)), "Value": vals})

    bars = (
        alt.Chart(df)
        .mark_bar(size=6)
        .encode(
            x=alt.X("Lag:Q", title="Lag"),
            y=alt.Y("Value:Q", title="PACF", scale=alt.Scale(domain=[-1, 1])),
            color=alt.condition(alt.datum.Value > 0, alt.value("#54A24B"), alt.value("#E45756")),
            tooltip=["Lag:Q", alt.Tooltip("Value:Q", format=".3f")],
        )
    )
    ci_df = pd.DataFrame({"y": [crit, -crit]})
    ci_lines = (
        alt.Chart(ci_df)
        .mark_rule(strokeDash=[4, 4], color="grey", opacity=0.8)
        .encode(y="y:Q")
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="black").encode(y="y:Q")
    return (bars + ci_lines + zero).properties(title=title, width="container", height=280).interactive()


def make_line_chart(df, x, y, title, color="#4C78A8", y_title=None):
    return (
        alt.Chart(df)
        .mark_line(color=color, strokeWidth=1.5)
        .encode(
            x=alt.X(f"{x}:T", title="Date"),
            y=alt.Y(f"{y}:Q", title=y_title or y),
            tooltip=[f"{x}:T", alt.Tooltip(f"{y}:Q", format=",.0f")],
        )
        .properties(title=title, width="container", height=260)
        .interactive()
    )


# 1. Revenue over time
st.title("Exploratory Data Analysis")

st.subheader("Weekly Revenue Over Time")
ts_df = df_weekly_ts.copy()
st.altair_chart(
    make_line_chart(ts_df, "Date", "Revenue", "Weekly Revenue – Online Retail II", y_title="Revenue (£)"),
    width='stretch',
)

# 2. ACF
st.subheader("Autocorrelation Function (ACF)")
st.markdown("""
- Gradual slow decay → series has memory / is non-stationary.
- Significant spikes at **lag 52** → strong annual seasonality (weekly data).
""")
lags = st.slider("ACF lags", 20, 80, 52, step=4)
st.altair_chart(make_acf_chart(series, lags, title="ACF of Weekly Revenue"), width='stretch')

# 3. PACF
st.subheader("Partial Autocorrelation Function (PACF)")
st.markdown("""
- Significant spikes indicate the AR order **p** for ARIMA/SARIMA.
- Grey dashed lines = 95% confidence bounds (±1.96/√n).
""")
lags_p = st.slider("PACF lags", 10, 52, 26, step=2)
st.altair_chart(make_pacf_chart(series, lags_p, title="PACF of Weekly Revenue"), width='stretch')

# 4. Seasonal Decomposition
st.subheader("Seasonal Decomposition")
st.markdown("""
Choose a model:
- **Additive** → seasonal swings have constant amplitude.
- **Multiplicative** → seasonal swings grow with the trend.
""")
model_type = st.radio("Decomposition model", ["additive", "multiplicative"], horizontal=True)

try:
    decomp = seasonal_decompose(series.dropna(), model=model_type, period=52)

    def decomp_chart(values, label, color):
        df_c = pd.DataFrame({"Date": values.index, "Value": values.values}).dropna()
        return (
            alt.Chart(df_c)
            .mark_line(color=color, strokeWidth=1.2)
            .encode(
                x=alt.X("Date:T", title=None),
                y=alt.Y("Value:Q", title=label),
                tooltip=["Date:T", alt.Tooltip("Value:Q", format=",.0f")],
            )
            .properties(title=label, width="container", height=160)
            .interactive()
        )

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    parts = [
        (series.dropna(), "Observed"),
        (decomp.trend,    "Trend"),
        (decomp.seasonal, "Seasonal"),
        (decomp.resid,    "Residual"),
    ]
    combined = alt.vconcat(*[decomp_chart(v, lbl, c) for (v, lbl), c in zip(parts, colors)])
    st.altair_chart(combined, width='stretch')

except Exception as e:
    st.error(f"Decomposition failed: {e}")

st.info("💡 A strong seasonal pattern at period=52 confirms annual retail cycles — key for choosing SARIMA / SARIMAX.")
