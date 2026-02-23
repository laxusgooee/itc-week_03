import streamlit as st

st.title("Business Insights")

st.markdown("""
## Overview

This analysis applied classical time series techniques to the **Online Retail II** dataset —
weekly revenue from a UK-based e-commerce retailer spanning 2009–2011.

The goal: understand revenue dynamics, validate stationarity, and produce reliable short-term forecasts.
""")

with st.expander("🔍 EDA & Seasonal Decomposition"):
    st.markdown("""
### Key Findings

- **Strong annual seasonality** (period ≈ 52 weeks) is visible in both the ACF and the seasonal decomposition.
- Revenue **peaks sharply in Q4** (October–December), driven by Christmas and holiday shopping.
- A **rising trend** is present across the dataset, indicating business growth over the period.
- The **multiplicative model** better captures the growing seasonal amplitude (swings get larger as revenue grows).

**Business Implication:**  
Stock, staffing, and logistics decisions should be planned around the Q4 surge,
with quieter periods (January–February) used for restocking and planning.
    """)

with st.expander("📊 Hypothesis Testing – Stationarity"):
    st.markdown("""
### ADF & KPSS Results

- The **raw series is non-stationary** — both the trend and seasonality violate the constant-mean assumption.
- After applying **d=1 (first differencing) + D=1 (seasonal differencing at lag 52)**,
  the ADF test rejects the unit root (p < 0.05) and the KPSS test fails to reject stationarity.
- This confirms the series is stationary at **I(1,1)** — first differenced, seasonally differenced.

**Model Selection Implication:**  
Use `d=1, D=1` for ARIMA/SARIMA orders. This is the standard for retail revenue time series.
    """)

with st.expander("📈 ARIMA Model"):
    st.markdown("""
### Performance

- **ARIMA(1,1,1)** provides a reasonable baseline, capturing short-term autocorrelation.
- The model performs well in stable periods but **misses seasonal peaks** — expected behaviour
  since ARIMA has no seasonal component.
- Residuals are approximately white noise for the non-seasonal period.

**Limitation:**  
ARIMA alone is insufficient for this dataset due to the strong annual cycle.
It should be viewed as a baseline to benchmark against SARIMA/SARIMAX.
    """)

with st.expander("📐 SARIMA Model"):
    st.markdown("""
### Performance

- **SARIMA(1,1,1)(1,1,1)[52]** significantly improves on ARIMA by explicitly modelling the 52-week cycle.
- The model captures both the **upward trend** and **holiday peaks**.
- AIC/BIC are lower than the equivalent ARIMA, confirming a better fit.
- Residuals show minimal autocorrelation — the seasonal pattern is well captured.

**Business Implication:**  
SARIMA forecasts are suitable for 4–12 week operational planning (inventory, staffing).
    """)

with st.expander("🔀 SARIMAX Model"):
    st.markdown("""
### Performance

- Adding **IsQ4** and **WeekOfYear** as exogenous variables further reduces AIC and tightens forecast intervals.
- The **IsQ4 coefficient is positive and significant**, confirming that Q4 generates materially higher revenue than the seasonal model alone accounts for.
- Month dummies (M10, M11, M12) systematically carry positive coefficients, validating the holiday effect.
- 95% confidence intervals are narrower than SARIMA, indicating exogenous features reduce residual variance.

**Business Implication:**  
SARIMAX is the preferred production model for this dataset. Marketing and procurement teams can use its
forecasts to plan promotions, optimise supply chains, and set revenue targets.
    """)

st.space()

st.subheader("Strategic Conclusion")

st.markdown("""
| Model   | Handles Trend | Handles Seasonality | Exogenous Vars | Best For |
|---------|:-------------:|:-------------------:|:--------------:|----------|
| ARIMA   | ✅            | ❌                  | ❌             | Baseline / short horizon |
| SARIMA  | ✅            | ✅                  | ❌             | Seasonal forecasting |
| SARIMAX | ✅            | ✅                  | ✅             | Production forecasting |

**Recommended approach for this retailer:**
1. Deploy **SARIMAX(1,1,1)(1,1,1)[52]** with IsQ4 + month dummies as the primary forecast engine.
2. Retrain monthly as new sales data arrives.
3. Use the 95% forecast interval to define **pessimistic / optimistic inventory scenarios**.
4. Investigate sharp residual spikes (unusually good or bad weeks) to identify promotional or external factors
   that can be added as future exogenous variables.
""")
