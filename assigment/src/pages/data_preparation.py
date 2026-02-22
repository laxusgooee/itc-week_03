import streamlit as st
import altair as alt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pages.data_summary import df_no_index as df

df = df.copy()

df_numeric = df[df.select_dtypes(include='number').columns]

# encode df #

# binary encoding
df["injury_prone"] = df["injury_prone"].map({"No": 0, "Yes": 1})
# ordinal encoding since it has an order
df["transfer_risk_level"] = df["transfer_risk_level"].map({"Low": 0, "Medium": 1, "High": 2})
# one hot encode
df = pd.get_dummies(df, columns=["position"], drop_first=False)
# remove categorical features
df_encoded = df.drop(columns=["club", "nationality"])

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)
df_scaled_encoded = scaler.fit_transform(df_encoded)

st.subheader("Encoding")
st.write(pd.DataFrame(df_scaled_encoded, columns=df_encoded.columns).head())

scaled = True

col_1, col_2 = st.columns([0.8, 0.2])

with col_1:
    st.subheader("Feature Scaling")

with col_2:
    scaled = st.toggle("With Scale", value=scaled)


frame = pd.DataFrame(df_scaled, columns=df_numeric.columns) if scaled else df_numeric

with st.container():
    st.write(frame.head())

st.space()

with st.expander("Show description"):
    st.write(frame.describe())

st.space()

with st.container():
    boxplot = alt.Chart(frame.melt(value_name="y", var_name="x")).mark_boxplot(extent='min-max').encode(
        x=alt.X('x:N', title='Features', scale=alt.Scale(zero=False)),
        y=alt.Y('y:Q', 
                title='Value', 
                scale=alt.Scale(zero=False)
            ),
        color=alt.Color('x', legend=None),
    ).properties(
        title="Box plot of all Numerical features",
        width='container',
        height=500
    ).interactive()

    st.altair_chart(boxplot, width='stretch')

