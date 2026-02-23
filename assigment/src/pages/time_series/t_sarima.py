import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf
from pages.time_series.t_data_utils import df_weekly_ts

series = df_weekly_ts.set_index("Date")["Revenue"]

st.title("SARIMA Model")
st.markdown("""
**SARIMA(p,d,q)(P,D,Q)ₛ** extends ARIMA with seasonal components:
- **(p, d, q)** – non-seasonal AR, differencing, MA
- **(P, D, Q)** – seasonal AR, differencing, MA
- **s = 52** – annual seasonality in weekly data
""")


# shared helpers
def acf_altair(s, nlags=26, title="ACF of Residuals"):
    n = len(s)
    vals = acf(s.dropna(), nlags=min(nlags, len(s.dropna()) - 1), fft=True)
    crit = 1.96 / np.sqrt(n)
    df = pd.DataFrame({"Lag": np.arange(len(vals)), "ACF": vals})
    bars = (
        alt.Chart(df).mark_bar(size=6)
        .encode(
            x="Lag:Q",
            y=alt.Y("ACF:Q", scale=alt.Scale(domain=[-1, 1])),
            color=alt.condition(alt.datum.ACF > 0, alt.value("#4C78A8"), alt.value("#E45756")),
            tooltip=["Lag:Q", alt.Tooltip("ACF:Q", format=".3f")],
        )
    )
    ci = alt.Chart(pd.DataFrame({"y": [crit, -crit]})).mark_rule(strokeDash=[4, 4], color="grey").encode(y="y:Q")
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="black").encode(y="y:Q")
    return (bars + ci + zero).properties(title=title, width="container", height=260).interactive()


# parameters 
st.subheader("Non-Seasonal Parameters")
c1, c2, c3 = st.columns(3)
p = int(c1.number_input("p", 0, 3, 1, key="sa_p"))
d = int(c2.number_input("d", 0, 2, 1, key="sa_d"))
q = int(c3.number_input("q", 0, 3, 1, key="sa_q"))

st.subheader("Seasonal Parameters  (s = 52)")
c4, c5, c6 = st.columns(3)
P = int(c4.number_input("P", 0, 2, 1, key="sa_P"))
D = int(c5.number_input("D", 0, 1, 1, key="sa_D"))
Q = int(c6.number_input("Q", 0, 2, 1, key="sa_Q"))
n_forecast = st.slider("Forecast horizon (weeks)", 4, 52, 12, key="sa_fc")

split = int(len(series) * 0.85)
train, test = series.iloc[:split], series.iloc[split:]

with st.spinner("Fitting SARIMA… ⏳"):
    try:
        result = SARIMAX(
            train, order=(p, d, q), seasonal_order=(P, D, Q, 52),
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)

        with st.expander("Model Summary"):
            st.text(result.summary().as_text())

        c1, c2, c3 = st.columns(3)
        c1.metric("AIC",  f"{result.aic:.2f}")
        c2.metric("BIC",  f"{result.bic:.2f}")
        c3.metric("HQIC", f"{result.hqic:.2f}")

        # fitted vs actual
        st.subheader("In-Sample Fit vs Actuals")
        fitted = result.fittedvalues
        df_fit = pd.concat([
            pd.DataFrame({"Date": train.index,  "Value": train.values,   "Series": "Train"}),
            pd.DataFrame({"Date": test.index,   "Value": test.values,    "Series": "Test (actual)"}),
            pd.DataFrame({"Date": fitted.index, "Value": fitted.values,  "Series": "Fitted"}),
        ])
        color_scale = alt.Scale(
            domain=["Train", "Test (actual)", "Fitted"],
            range=["#4C78A8", "#F58518", "#54A24B"],
        )
        st.altair_chart(
            alt.Chart(df_fit).mark_line(strokeWidth=1.4).encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("Value:Q", title="Revenue (£)"),
                color=alt.Color("Series:N", scale=color_scale),
                strokeDash=alt.condition(alt.datum.Series == "Fitted", alt.value([4, 2]), alt.value([0])),
                tooltip=["Date:T", "Series:N", alt.Tooltip("Value:Q", format=",.0f")],
            )
            .properties(title=f"SARIMA({p},{d},{q})({P},{D},{Q})[52] – In-Sample Fit", width="container", height=320)
            .interactive(),
            width='stretch',
        )

        # forecast
        st.subheader(f"{n_forecast}-Week Forecast")
        fc   = result.get_forecast(steps=n_forecast)
        mean = fc.predicted_mean
        ci   = fc.conf_int(alpha=0.05)

        recent   = series.iloc[-60:]
        df_recent = pd.DataFrame({"Date": recent.index, "Value": recent.values})
        df_fc    = pd.DataFrame({"Date": mean.index, "Value": mean.values,
                                 "Lower": ci.iloc[:, 0].values, "Upper": ci.iloc[:, 1].values})
        band     = alt.Chart(df_fc).mark_area(opacity=0.2, color="#E45756").encode(x="Date:T", y="Lower:Q", y2="Upper:Q")
        act_line = alt.Chart(df_recent).mark_line(color="#4C78A8", strokeWidth=1.4).encode(
                       x="Date:T", y=alt.Y("Value:Q", title="Revenue (£)"),
                       tooltip=["Date:T", alt.Tooltip("Value:Q", format=",.0f")])
        fc_line  = alt.Chart(df_fc).mark_line(color="#E45756", strokeWidth=2).encode(
                       x="Date:T", y="Value:Q", tooltip=["Date:T", alt.Tooltip("Value:Q", format=",.0f")])
        st.altair_chart(
            (band + act_line + fc_line)
            .properties(title=f"SARIMA({p},{d},{q})({P},{D},{Q})[52] – Forecast", width="container", height=320)
            .interactive(),
            width='stretch',
        )

        # residuals
        st.subheader("Residual Diagnostics")
        resid = result.resid.dropna()
        c_left, c_right = st.columns(2)
        with c_left:
            st.altair_chart(acf_altair(resid), width='stretch')
        with c_right:
            resid_df = pd.DataFrame({"Residual": resid.values})
            st.altair_chart(
                alt.Chart(resid_df).mark_bar(color="#E45756", opacity=0.8).encode(
                    x=alt.X("Residual:Q", bin=alt.Bin(maxbins=30)),
                    y="count()",
                    tooltip=["count()"],
                ).properties(title="Residual Distribution", width="container", height=260),
                width='stretch',
            )

    except Exception as e:
        st.error(f"SARIMA fitting failed: {e}")
        st.info("Try reducing P, D, Q or using D=0 if the series is short.")
