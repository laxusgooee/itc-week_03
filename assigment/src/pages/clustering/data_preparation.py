import streamlit as st
import altair as alt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pages.clustering.data_summary import df_no_index as df

df = df.copy()

df_numeric = df[df.select_dtypes(include='number').columns]

# Encoding
st.title("Data Preparation")

st.subheader("Step 1 – Encoding")
st.markdown("""
To use the data in machine learning algorithms, categorical columns must be converted to numbers:

| Column | Method | Rationale |
|--------|--------|-----------|
| `injury_prone` | Binary (No→0, Yes→1) | Only two categories |
| `transfer_risk_level` | Ordinal (Low→0, Medium→1, High→2) | Natural ordering |
| `position` | One-hot encoding | No ordinal relationship |
| `club`, `nationality` | Dropped | Too many categories, low clustering signal |
""")

df["injury_prone"] = df["injury_prone"].map({"No": 0, "Yes": 1})
df["transfer_risk_level"] = df["transfer_risk_level"].map({"Low": 0, "Medium": 1, "High": 2})
df = pd.get_dummies(df, columns=["position"], drop_first=False)
df_encoded = df.drop(columns=["club", "nationality"])

st.write(df_encoded.head())

# Scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)
df_scaled_encoded = scaler.fit_transform(df_encoded)

scaled = True

col_1, col_2 = st.columns([0.8, 0.2])
with col_1:
    st.subheader("Step 2 – Feature Scaling")
    st.markdown("""
StandardScaler transforms each feature to **zero mean** and **unit variance**:
`z = (x - μ) / σ`

This prevents features with large ranges (e.g., `market_value_euros`) from dominating
the distance calculations in K-Means.
    """)
with col_2:
    scaled = st.toggle("With Scale", value=scaled)

frame = pd.DataFrame(df_scaled, columns=df_numeric.columns) if scaled else df_numeric

with st.container():
    st.write(frame.head())

st.space()

with st.expander("Show description"):
    st.write(frame.describe())

st.space()

# Box plot
st.subheader("Step 3 – Feature Distribution")
st.markdown("The box plot below shows how scaling affects the feature distributions.")

with st.container():
    boxplot = (
        alt.Chart(frame.melt(value_name="y", var_name="x"))
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("x:N", title="Features", scale=alt.Scale(zero=False)),
            y=alt.Y("y:Q", title="Value", scale=alt.Scale(zero=False)),
            color=alt.Color("x", legend=None),
        )
        .properties(
            title="Box Plot of All Numerical Features",
            width="container",
            height=500,
        )
        .interactive()
    )
    st.altair_chart(boxplot, width='stretch')

# Variance after scaling
st.subheader("Step 4 – Variance per Feature")
variance_df = pd.DataFrame({
    "Feature":  frame.columns,
    "Variance": frame.var().values,
})
var_chart = (
    alt.Chart(variance_df)
    .mark_bar(color="#4C78A8", opacity=0.85)
    .encode(
        x=alt.X("Feature:N", sort="-y", title="Feature"),
        y=alt.Y("Variance:Q", title="Variance"),
        tooltip=["Feature", alt.Tooltip("Variance:Q", format=".3f")],
    )
    .properties(title="Feature Variance (after scaling)", width="container", height=300)
    .interactive()
)
st.altair_chart(var_chart, width='stretch')
