"""
Shared time-series data layer.
Import from this module in any downstream page – no st.* calls here.
"""
import pandas as pd
from pages.time_series.t_data_summary import df_no_index as _raw

# clean
df = _raw.copy()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df = df[df["Price"] > 0]
df["Revenue"] = df["Quantity"] * df["Price"]

# daily aggregation
df_daily_ts = (
    df.groupby(df["InvoiceDate"].dt.date)
    .agg(
        TotalRevenue=("Revenue", "sum"),
        TotalOrders=("Invoice", "nunique"),
        TotalItems=("Quantity", "sum"),
    )
    .reset_index()
    .rename(columns={"InvoiceDate": "Date"})
)
df_daily_ts["Date"] = pd.to_datetime(df_daily_ts["Date"])
df_daily_ts = df_daily_ts.sort_values("Date").reset_index(drop=True)

# weekly aggregation
df_weekly_ts = (
    df_daily_ts.set_index("Date")["TotalRevenue"]
    .resample("W")
    .sum()
    .reset_index()
    .rename(columns={"TotalRevenue": "Revenue"})
)

# exogenous features (for SARIMAX)
def make_exog(ts: pd.DataFrame) -> pd.DataFrame:
    """Add month dummies and week-of-year to a weekly ts DataFrame."""
    ts = ts.copy()
    ts["Month"] = ts["Date"].dt.month
    ts["WeekOfYear"] = ts["Date"].dt.isocalendar().week.astype(int)
    ts["IsQ4"] = ts["Month"].isin([10, 11, 12]).astype(int)
    month_dummies = pd.get_dummies(ts["Month"], prefix="M", drop_first=True)
    return pd.concat([ts, month_dummies], axis=1).drop(columns=["Month"])
