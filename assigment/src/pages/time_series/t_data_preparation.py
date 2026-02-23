import streamlit as st
import altair as alt
import pandas as pd
from pages.time_series.t_data_utils import df_daily_ts, df_weekly_ts

# 1. Revenue Feature Overview
st.title("Data Preparation")

st.subheader("Step 1 – Revenue Derivation")
st.markdown("""
From the raw transactional data we derive:

| Column | Formula | Notes |
|--------|---------|-------|
| `Revenue` | `Quantity × Price` | Negative when Quantity < 0 (returns) |
| `Date` | `InvoiceDate.dt.date` | Date-only (no time) |

Rows with `Price ≤ 0` are **removed** (adjustments / samples).
""")

col1, col2, col3 = st.columns(3, border=True)
col1.metric("Daily observations", f"{len(df_daily_ts):,}")
col2.metric("Weekly observations", f"{len(df_weekly_ts):,}")
col3.metric("Date range",
            f"{df_daily_ts['Date'].min().strftime('%Y-%m')} → "
            f"{df_daily_ts['Date'].max().strftime('%Y-%m')}")

# 2. Daily time series table
st.subheader("Step 2 – Daily Aggregation")
st.markdown("""
Transactions are grouped by `Date` to compute:
- `TotalRevenue` – sum of all Revenue
- `TotalOrders` – unique invoices
- `TotalItems` – sum of Quantity
""")
st.write(df_daily_ts.head(10))

with st.expander("Descriptive statistics"):
    st.write(df_daily_ts.describe())

# 3. Resampling
st.subheader("Step 3 – Resampling")
freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
freq_label = st.selectbox("Aggregation frequency", list(freq_map.keys()), index=1)
freq = freq_map[freq_label]

resampled = (
    df_daily_ts.set_index("Date")["TotalRevenue"]
    .resample(freq)
    .sum()
    .reset_index()
    .rename(columns={"TotalRevenue": "Revenue"})
)
st.write(resampled.head(10))

# 4. Revenue trend chart
st.subheader("Step 4 – Revenue Trend")

base  = alt.Chart(resampled).encode(x=alt.X("Date:T", title="Date"))
line  = base.mark_line(color="#4C78A8").encode(
    y=alt.Y("Revenue:Q", title="Revenue (£)"), tooltip=["Date:T", "Revenue:Q"]
)
pts   = base.mark_point(filled=True, size=40, color="#4C78A8").encode(y="Revenue:Q")

chart = (line + pts).properties(
    title=f"{freq_label} Revenue Trend – Online Retail II",
    width="container", height=400,
).interactive()

st.altair_chart(chart, width='stretch')
