import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf
from pages.time_series.t_data_utils import df_weekly_ts

series = df_weekly_ts.set_index("Date")["Revenue"]

st.title("ARIMA Model")
st.markdown("""
**ARIMA(p, d, q)** — AutoRegressive Integrated Moving Average:
- **p** – AR order (past values)
- **d** – Differencing order (stationarity)
- **q** – MA order (past errors)

ARIMA does **not** model seasonal patterns — see SARIMA for that.
""")


# shared Altair helpers
def acf_altair(s, nlags=26, title="ACF of Residuals"):
    n = len(s)
    vals = acf(s.dropna(), nlags=min(nlags, len(s.dropna()) - 1), fft=True)
    crit = 1.96 / np.sqrt(n)
    df = pd.DataFrame({"Lag": np.arange(len(vals)), "ACF": vals})
    bars = (
        alt.Chart(df).mark_bar(size=6)
        .encode(
            x=alt.X("Lag:Q"),
            y=alt.Y("ACF:Q", scale=alt.Scale(domain=[-1, 1])),
            color=alt.condition(alt.datum.ACF > 0, alt.value("#4C78A8"), alt.value("#E45756")),
            tooltip=["Lag:Q", alt.Tooltip("ACF:Q", format=".3f")],
        )
    )
    ci = alt.Chart(pd.DataFrame({"y": [crit, -crit]})).mark_rule(strokeDash=[4, 4], color="grey").encode(y="y:Q")
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="black").encode(y="y:Q")
    return (bars + ci + zero).properties(title=title, width="container", height=260).interactive()


# parameters 
st.subheader("Model Parameters")
c1, c2, c3 = st.columns(3)
p = int(c1.number_input("p  (AR order)",    0, 5, 1))
d = int(c2.number_input("d  (differencing)", 0, 2, 1))
q = int(c3.number_input("q  (MA order)",    0, 5, 1))
n_forecast = st.slider("Forecast horizon (weeks)", 4, 52, 12)

split = int(len(series) * 0.85)
train, test = series.iloc[:split], series.iloc[split:]

# fit
with st.spinner("Fitting ARIMA…"):
    try:
        result = ARIMA(train, order=(p, d, q)).fit()

        with st.expander("Model Summary"):
            st.text(result.summary().as_text())

        c1, c2, c3 = st.columns(3)
        c1.metric("AIC",  f"{result.aic:.2f}")
        c2.metric("BIC",  f"{result.bic:.2f}")
        c3.metric("HQIC", f"{result.hqic:.2f}")

        # in-sample fit
        st.subheader("In-Sample Fit vs Actuals")
        fitted = result.fittedvalues
        df_fit = pd.concat([
            pd.DataFrame({"Date": train.index, "Value": train.values, "Series": "Train"}),
            pd.DataFrame({"Date": test.index,  "Value": test.values,  "Series": "Test (actual)"}),
            pd.DataFrame({"Date": fitted.index, "Value": fitted.values, "Series": "Fitted"}),
        ])
        color_scale = alt.Scale(
            domain=["Train", "Test (actual)", "Fitted"],
            range=["#4C78A8", "#F58518", "#54A24B"],
        )
        fit_chart = (
            alt.Chart(df_fit)
            .mark_line(strokeWidth=1.4)
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("Value:Q", title="Revenue (£)"),
                color=alt.Color("Series:N", scale=color_scale),
                strokeDash=alt.condition(
                    alt.datum.Series == "Fitted",
                    alt.value([4, 2]),
                    alt.value([0]),
                ),
                tooltip=["Date:T", "Series:N", alt.Tooltip("Value:Q", format=",.0f")],
            )
            .properties(title=f"ARIMA({p},{d},{q}) – In-Sample Fit", width="container", height=320)
            .interactive()
        )
        st.altair_chart(fit_chart, width='stretch')

        # forecast
        st.subheader(f"Forecast – next {n_forecast} weeks")
        fc_obj = result.get_forecast(steps=n_forecast)
        mean   = fc_obj.predicted_mean
        ci     = fc_obj.conf_int(alpha=0.05)

        recent = series.iloc[-52:]
        df_recent = pd.DataFrame({"Date": recent.index, "Value": recent.values})
        df_fc = pd.DataFrame({
            "Date":  mean.index,
            "Value": mean.values,
            "Lower": ci.iloc[:, 0].values,
            "Upper": ci.iloc[:, 1].values,
        })
        band = (
            alt.Chart(df_fc).mark_area(opacity=0.2, color="#E45756")
            .encode(x="Date:T", y="Lower:Q", y2="Upper:Q")
        )
        actual_line = (
            alt.Chart(df_recent).mark_line(color="#4C78A8", strokeWidth=1.4)
            .encode(x="Date:T", y=alt.Y("Value:Q", title="Revenue (£)"),
                    tooltip=["Date:T", alt.Tooltip("Value:Q", format=",.0f")])
        )
        fc_line = (
            alt.Chart(df_fc).mark_line(color="#E45756", strokeWidth=2)
            .encode(x="Date:T", y="Value:Q", tooltip=["Date:T", alt.Tooltip("Value:Q", format=",.0f")])
        )
        st.altair_chart(
            (band + actual_line + fc_line)
            .properties(title=f"ARIMA({p},{d},{q}) – {n_forecast}-Week Forecast", width="container", height=320)
            .interactive(),
            width='stretch',
        )

        # residual diagnostics
        st.subheader("Residual Diagnostics")
        resid = result.resid.dropna()

        c_left, c_right = st.columns(2)
        with c_left:
            st.altair_chart(acf_altair(resid, title="ACF of Residuals"), width='stretch')
        with c_right:
            resid_df = pd.DataFrame({"Residual": resid.values})
            hist = (
                alt.Chart(resid_df)
                .mark_bar(color="#4C78A8", opacity=0.8)
                .encode(
                    x=alt.X("Residual:Q", bin=alt.Bin(maxbins=30), title="Residual"),
                    y=alt.Y("count()", title="Count"),
                    tooltip=["count()"],
                )
                .properties(title="Residual Distribution", width="container", height=260)
            )
            st.altair_chart(hist, width='stretch')

    except Exception as e:
        st.error(f"Model fitting failed: {e}")
