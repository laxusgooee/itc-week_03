import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from pages.time_series.t_data_utils import df_weekly_ts

series = df_weekly_ts.set_index("Date")["Revenue"]

st.title("Hypothesis Testing – Stationarity")

st.markdown("""
Most time series models assume **stationarity** — statistical properties do not change over time.

| Test | H₀ (null hypothesis) | Reject H₀ when |
|------|----------------------|----------------|
| **ADF** |  There is no stationarity in the data | p-value < 0.05 |
| **KPSS** | Series is stationary | p-value < 0.05 |
""")

# helpers
def interpret_adf(p):
    if p < 0.01:   return "✅ Strongly stationary (p < 0.01)"
    elif p < 0.05: return "✅ Stationary (p < 0.05)"
    elif p < 0.10: return "⚠️ Weakly stationary (p < 0.10)"
    else:          return "❌ Non-stationary (p ≥ 0.05)"

def interpret_kpss(p):
    if p > 0.10:   return "✅ Stationary (p > 0.10)"
    elif p > 0.05: return "✅ Likely stationary (p > 0.05)"
    else:          return "❌ Non-stationary (p ≤ 0.05)"

# differencing controls
st.subheader("Apply Differencing")
d = st.selectbox("Differencing order (d)", [0, 1, 2], index=1,
                 help="d=0 raw | d=1 first difference | d=2 second difference")
D = st.selectbox("Seasonal differencing (D) at period 52", [0, 1], index=1)

s = series.dropna().copy()
if D == 1:
    s = s.diff(52).dropna()
if d > 0:
    s = s.diff(d).dropna()

# differenced series chart
df_plot = pd.DataFrame({"Date": s.index, "Revenue": s.values})
line_chart = (
    alt.Chart(df_plot)
    .mark_line(color="#4C78A8", strokeWidth=1.3)
    .encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Revenue:Q", title="Revenue (£)"),
        tooltip=["Date:T", alt.Tooltip("Revenue:Q", format=",.0f")],
    )
    .properties(
        title=f"Series after d={d}, D={D} (period=52) differencing",
        width="container",
        height=280,
    )
    .interactive()
)
st.altair_chart(line_chart, width='stretch')

# ADF Test
st.subheader("Augmented Dickey-Fuller (ADF) Test")

adf_stat, adf_p, adf_lags, _, adf_crit, _ = adfuller(s, autolag="AIC")


col1, col2, col3 = st.columns(3)
col1.metric("ADF Statistic", f"{adf_stat:.4f}")
col2.metric("p-value",       f"{adf_p:.4f}")
col3.metric("Lags used",     adf_lags)
st.markdown(f"**Interpretation:** {interpret_adf(adf_p)}")

with st.expander("ADF Critical values"):
    st.table(pd.DataFrame(adf_crit, index=["Critical Value"]).T.rename_axis("Significance"))

# KPSS Test 
st.subheader("KPSS Test")
kpss_p = None
try:
    kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(s, regression="c", nlags="auto")

    col1, col2, col3 = st.columns(3)
    col1.metric("KPSS Statistic", f"{kpss_stat:.4f}")
    col2.metric("p-value",        f"{kpss_p:.4f}")
    col3.metric("Lags used",      kpss_lags)
    st.markdown(f"**Interpretation:** {interpret_kpss(kpss_p)}")

    with st.expander("KPSS Critical values"):
        st.table(pd.DataFrame(kpss_crit, index=["Critical Value"]).T.rename_axis("Significance"))
except Exception as e:
    st.error(f"KPSS failed: {e}")

# Stationarity bar summary
st.subheader("Test Results at a Glance")

results_df = pd.DataFrame({
    "Test": ["ADF", "KPSS"],
    "p-value": [round(adf_p, 4), round(kpss_p, 4) if kpss_p is not None else float("nan")],
    "Stationary?": [
        "Yes" if adf_p < 0.05 else "No",
        "Yes" if (kpss_p is not None and kpss_p > 0.05) else "No",
    ],
})

bar = (
    alt.Chart(results_df)
    .mark_bar()
    .encode(
        x=alt.X("Test:N", title=None),
        y=alt.Y("p-value:Q", scale=alt.Scale(domain=[0, max(0.5, results_df["p-value"].max() + 0.05)])),
        color=alt.Color(
            "Stationary?:N",
            scale=alt.Scale(domain=["Yes", "No"], range=["#54A24B", "#E45756"]),
        ),
        tooltip=["Test:N", alt.Tooltip("p-value:Q", format=".4f"), "Stationary?:N"],
    )
    .properties(title="p-value by Test (green = stationary)", width="container", height=260)
)
threshold = (
    alt.Chart(pd.DataFrame({"y": [0.05]}))
    .mark_rule(strokeDash=[4, 4], color="grey")
    .encode(y="y:Q")
)
st.altair_chart(bar + threshold, width='stretch')

# Conclusion 
st.subheader("Conclusion")
adf_ok  = adf_p < 0.05
kpss_ok = kpss_p is not None and kpss_p > 0.05

if adf_ok and kpss_ok:
    st.success(f"Both tests agree: the series is **stationary** with d={d}, D={D}. Ready for ARIMA/SARIMA/SARIMAX.")
elif adf_ok and not kpss_ok:
    st.warning("Mixed signals — ADF suggests stationarity but KPSS does not. Consider additional differencing.")
elif not adf_ok and kpss_ok:
    st.warning("Mixed signals — KPSS suggests stationarity but ADF does not.")
else:
    st.error("Both tests indicate **non-stationarity**. Increase differencing order.")
