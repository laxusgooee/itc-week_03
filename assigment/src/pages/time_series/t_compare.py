import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from pages.time_series.t_data_utils import df_weekly_ts, make_exog

# data
series      = df_weekly_ts.set_index("Date")["Revenue"]
df_exog_all = make_exog(df_weekly_ts).set_index("Date")

st.title("Model Comparison: ARIMA vs SARIMA vs SARIMAX")
st.markdown("""
Compare the three models on the **same train / test split** using held-out test data.
Metrics are computed on the **test set** (last 15 % of weeks).

| Metric | Formula | Lower = better |
|--------|---------|:--------------:|
| **MAE** | mean |y - ŷ| | ✅ |
| **RMSE** | √mean(y - ŷ)² | ✅ |
| **MAPE** | mean |y - ŷ| / |y| × 100 | ✅ |
| **R²** | 1 - SS_res / SS_tot | ❌ (higher = better) |
""")

# shared parameters
st.subheader("Shared Parameters")
col1, col2, col3 = st.columns(3)
p = int(col1.number_input("p", 0, 3, 1, key="cmp_p"))
d = int(col2.number_input("d", 0, 2, 1, key="cmp_d"))
q = int(col3.number_input("q", 0, 3, 1, key="cmp_q"))

col4, col5, col6 = st.columns(3)
P = int(col4.number_input("P (seasonal)", 0, 2, 1, key="cmp_P"))
D = int(col5.number_input("D (seasonal)", 0, 1, 1, key="cmp_D"))
Q = int(col6.number_input("Q (seasonal)", 0, 2, 1, key="cmp_Q"))

exog_cols   = [c for c in df_exog_all.columns if c not in ["Revenue"]]
chosen_exog = st.multiselect("Exogenous features (SARIMAX only)", exog_cols,
                              default=["IsQ4", "WeekOfYear"], key="cmp_exog")

# split
split   = int(len(series) * 0.85)
train   = series.iloc[:split]
test    = series.iloc[split:]
n_test  = len(test)

e_train = df_exog_all[chosen_exog].iloc[:split]  if chosen_exog else None
e_test  = df_exog_all[chosen_exog].iloc[split:]  if chosen_exog else None

st.caption(f"Train: **{len(train)}** weeks  |  Test: **{n_test}** weeks")

# helpers
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R²":   r2_score(y_true, y_pred),
    }

#fit all three models ────
results   = {}   # model_name -> statsmodels result
forecasts = {}   # model_name -> predicted Series aligned to test index
errors    = {}   # model_name -> error string

with st.spinner("Fitting all three models… ⏳"):

    # ARIMA
    try:
        r = ARIMA(train, order=(p, d, q)).fit()
        fc = r.get_forecast(steps=n_test).predicted_mean
        fc.index = test.index
        results["ARIMA"]   = r
        forecasts["ARIMA"] = fc
    except Exception as e:
        errors["ARIMA"] = str(e)

    # SARIMA
    try:
        r = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, 52),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = r.get_forecast(steps=n_test).predicted_mean
        fc.index = test.index
        results["SARIMA"]   = r
        forecasts["SARIMA"] = fc
    except Exception as e:
        errors["SARIMA"] = str(e)

    # SARIMAX
    try:
        r = SARIMAX(train, exog=e_train, order=(p, d, q), seasonal_order=(P, D, Q, 52),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = r.get_forecast(steps=n_test, exog=e_test).predicted_mean
        fc.index = test.index
        results["SARIMAX"]   = r
        forecasts["SARIMAX"] = fc
    except Exception as e:
        errors["SARIMAX"] = str(e)

for name, msg in errors.items():
    st.warning(f"**{name}** failed: {msg}")

if not forecasts:
    st.stop()

#metrics table 
st.subheader("📊 Test-Set Metrics")
metric_rows = []
for name, fc in forecasts.items():
    m = compute_metrics(test.values, fc.values)
    metric_rows.append({"Model": name, **m})

metric_df = pd.DataFrame(metric_rows)

# highlight best per metric
def highlight_best(col):
    if col.name == "R²":
        best = col.max()
        return ["background-color:#d4edda" if v == best else "" for v in col]
    else:
        best = col.min()
        return ["background-color:#d4edda" if v == best else "" for v in col]

styled = (
    metric_df.style
    .apply(highlight_best, subset=["MAE", "RMSE", "MAPE", "R²"])
    .format({"MAE": "£{:,.0f}", "RMSE": "£{:,.0f}", "MAPE": "{:.2f}%", "R²": "{:.4f}"})
)
st.dataframe(styled, use_container_width=True, hide_index=True)

#metric bar charts ───────
st.subheader("📉 Metric Comparison Charts")

palette = alt.Scale(
    domain=["ARIMA", "SARIMA", "SARIMAX"],
    range=["#4C78A8", "#F58518", "#54A24B"],
)

def bar_chart(df, metric, title, fmt=None):
    base = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Model:N", axis=alt.Axis(labelAngle=0), title=None),
            y=alt.Y(f"{metric}:Q", title=metric),
            color=alt.Color("Model:N", scale=palette, legend=None),
            tooltip=["Model:N", alt.Tooltip(f"{metric}:Q", format=fmt or ",.2f")],
        )
        .properties(title=title, width="container", height=220)
    )
    return base

c1, c2 = st.columns(2)
with c1:
    st.altair_chart(bar_chart(metric_df, "MAE",  "MAE (£ — lower is better)", ",.0f"), width='stretch')
    st.altair_chart(bar_chart(metric_df, "MAPE", "MAPE % (lower is better)",  ".2f"),  width='stretch')
with c2:
    st.altair_chart(bar_chart(metric_df, "RMSE", "RMSE (£ — lower is better)", ",.0f"), width='stretch')
    st.altair_chart(bar_chart(metric_df, "R²",   "R² (higher is better)",       ".4f"),  width='stretch')

#overlay: actual vs all forecasts ───
st.subheader("🔍 Forecast Overlay – Test Period")

# recent actual (last 40 weeks of train + full test)
lookback   = min(40, len(train))
recent_act = series.iloc[split - lookback:]

rows = [pd.DataFrame({"Date": recent_act.index, "Value": recent_act.values, "Series": "Actual"})]
for name, fc in forecasts.items():
    rows.append(pd.DataFrame({"Date": fc.index, "Value": fc.values, "Series": name}))
df_overlay = pd.concat(rows, ignore_index=True)

overlay_palette = alt.Scale(
    domain=["Actual", "ARIMA", "SARIMA", "SARIMAX"],
    range=["#AAAAAA", "#4C78A8", "#F58518", "#54A24B"],
)
overlay_dash = alt.condition(
    alt.datum.Series == "Actual",
    alt.value([0]),
    alt.value([5, 3]),
)

overlay_chart = (
    alt.Chart(df_overlay)
    .mark_line(strokeWidth=2)
    .encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Value:Q", title="Revenue (£)"),
        color=alt.Color("Series:N", scale=overlay_palette),
        strokeDash=overlay_dash,
        tooltip=["Date:T", "Series:N", alt.Tooltip("Value:Q", format=",.0f")],
    )
    .properties(title="Actual vs Forecast – Test Set", width="container", height=360)
    .interactive()
)

# st.altair_chart(overlay_chart, width='stretch')

# vertical divider line at split date
split_date = pd.DataFrame({"Date": [train.index[-1]]})
vline = (
    alt.Chart(split_date)
    .mark_rule(color="black", strokeDash=[4, 4], strokeWidth=1)
    .encode(x="Date:T", tooltip=alt.value("Train / Test split"))
)
st.altair_chart((overlay_chart + vline).properties(height=0), width='stretch')

#AIC / BIC comparison ───
st.subheader("🧪 Information Criteria (train-set fit quality)")
ic_rows = []
for name, r in results.items():
    ic_rows.append({"Model": name, "AIC": r.aic, "BIC": r.bic, "HQIC": r.hqic})
ic_df = pd.DataFrame(ic_rows)

c1, c2 = st.columns(2)
with c1:
    st.altair_chart(bar_chart(ic_df, "AIC",  "AIC (lower is better)", ",.1f"), width='stretch')
with c2:
    st.altair_chart(bar_chart(ic_df, "BIC",  "BIC (lower is better)", ",.1f"), width='stretch')

st.dataframe(
    ic_df.style.format({"AIC": "{:,.2f}", "BIC": "{:,.2f}", "HQIC": "{:,.2f}"}),
    use_container_width=True, hide_index=True,
)
