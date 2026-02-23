import streamlit as st
import altair as alt
import pandas as pd
from data import df_time_series as df

@st.dialog("Data types")
def show_types():
    st.write(df.dtypes)

st.title("Our Data")

# pipeline workflow
_steps = [
    "Data Summary", "Data Preparation", "EDA",
    "Hypothesis Testing", "ARIMA", "SARIMA", "SARIMAX", "Business Insights",
]
_current = "Data Summary"
_pipe_df = pd.DataFrame({
    "x":     list(range(len(_steps))),
    "y":     [0] * len(_steps),
    "label": _steps,
    "active": [s == _current for s in _steps],
})
_nodes = (
    alt.Chart(_pipe_df)
    .mark_circle(size=800)
    .encode(
        x=alt.X("x:O", axis=None),
        y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-0.5, 0.5])),
        color=alt.condition(
            alt.datum.active,
            alt.value("#E45756"),
            alt.value("#F5B5B4"),
        ),
        tooltip=["label:N"],
    )
)
_labels = (
    alt.Chart(_pipe_df)
    .mark_text(dy=32, fontSize=10, fontWeight="bold")
    .encode(
        x=alt.X("x:O", axis=None),
        y=alt.Y("y:Q", axis=None),
        text=alt.Text("label:N"),
        color=alt.condition(
            alt.datum.active,
            alt.value("#E45756"),
            alt.value("#888888"),
        ),
    )
)
_edge_rows = [
    {"x": i + 0.15, "x2": i + 0.85, "y": 0, "y2": 0}
    for i in range(len(_steps) - 1)
]
_edges_df = pd.DataFrame(_edge_rows)
_edges = (
    alt.Chart(_edges_df)
    .mark_rule(color="#F5B5B4", strokeWidth=2)
    .encode(
        x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-0.5, len(_steps) - 0.5])),
        x2="x2:Q",
        y=alt.Y("y:Q", axis=None),
        y2="y2:Q",
    )
)
_workflow = (
    (_edges + _nodes + _labels)
    .properties(width="container", height=90)
    .configure_view(strokeWidth=0)
)
st.altair_chart(_workflow, use_container_width=True)

st.divider()

#top metrics ──

null_count = int(df.isna().sum().sum())
dup_count  = int(df.duplicated().sum())

with st.container():
    col_1, col_2, col_3, col_4 = st.columns(4, border=True)

    with col_1:
        st.subheader("Shape:")
        st.header(str(df.shape))

    with col_2:
        st.subheader("Null Values:")
        st.header(null_count)

    with col_3:
        st.subheader("Duplicates:")
        st.header(dup_count)

    with col_4:
        st.subheader("Features:")
        st.header(df.shape[1])

#drop duplicates (after showing the count above) ─

df = df.drop_duplicates()
df_no_index = df.drop(["Customer ID", "Country"], axis=1)

#data preview ─

with st.container():
    show_indexes = True
    col_1, col_2 = st.columns([0.7, 0.3], vertical_alignment="center")

    with col_1:
        st.subheader("Data Sample")
    with col_2:
        with st.container(height="content", vertical_alignment="bottom", horizontal_alignment="right"):
            show_indexes = st.toggle(
                "Hide Indexes",
                value=show_indexes,
                help="Hides Customer ID and Country columns",
            )

    st.write(df_no_index.head() if show_indexes else df.head())

#description + types ──────

with st.container():
    col_1, col_2 = st.columns([0.6, 0.4], vertical_alignment="bottom")

    with col_1:
        st.subheader("Data Description")
    with col_2:
        with st.container(horizontal_alignment="right"):
            st.button("Show types", on_click=show_types)

    st.write(df_no_index.describe())

st.divider()

# missing values 

null_series = df.isna().sum()
null_series = null_series[null_series > 0]

if len(null_series) > 0:
    st.subheader("Missing Values per Feature")
    null_df = pd.DataFrame({"Feature": null_series.index, "Missing": null_series.values})
    null_chart = (
        alt.Chart(null_df)
        .mark_bar(color="#E45756")
        .encode(
            x=alt.X("Feature:N", sort="-y"),
            y=alt.Y("Missing:Q", title="Missing Count"),
            tooltip=["Feature", alt.Tooltip("Missing:Q", format=",")],
        )
        .properties(title="Missing Values per Column", width="container", height=280)
    )
    st.altair_chart(null_chart, use_container_width=True)
else:
    st.success("✅ No missing values detected in the dataset.")

#feature type breakdown ───

st.subheader("Feature Type Breakdown")
dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
dtype_counts.columns = ["Type", "Count"]
dtype_chart = (
    alt.Chart(dtype_counts)
    .mark_arc(innerRadius=50)
    .encode(
        theta=alt.Theta("Count:Q"),
        color=alt.Color("Type:N", scale=alt.Scale(scheme="tableau10")),
        tooltip=["Type", "Count"],
    )
    .properties(title="Data Types", width=320, height=280)
)
st.altair_chart(dtype_chart, use_container_width=False)
