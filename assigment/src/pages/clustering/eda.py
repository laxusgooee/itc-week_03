import streamlit as st
import altair as alt
import pandas as pd
from pages.clustering.data_summary import df_no_index as df

st.title("EDA")

# 1. Age Distribution
st.subheader("Age Distribution")
age_chart = (
    alt.Chart(df)
    .mark_bar(color="#4C78A8", opacity=0.85)
    .encode(
        alt.X("age", bin=alt.Bin(maxbins=20), title="Age"),
        alt.Y("count()", title="Count"),
        tooltip=[alt.Tooltip("age", bin=True), "count()"],
    )
    .properties(title="Age Distribution of Professional Football Players", width="container", height=300)
    .interactive()
)
st.altair_chart(age_chart, use_container_width=True)

# 2. Rating vs Age by Position (interactive legend)
st.subheader("Overall Rating vs Age")
selection = alt.selection_point(fields=["position"], bind="legend")
scatter = (
    alt.Chart(df)
    .mark_circle(size=55, opacity=0.7)
    .encode(
        x=alt.X("age:Q", title="Age", scale=alt.Scale(zero=False)),
        y=alt.Y("overall_rating:Q", title="Overall Rating", scale=alt.Scale(zero=False)),
        color=alt.Color("position:N", title="Position"),
        tooltip=["age", "overall_rating", "position"],
        opacity=alt.condition(selection, alt.value(0.75), alt.value(0.05)),
    )
    .add_params(selection)
    .properties(title="Impact of Age on Overall Rating – click legend to filter", width="container", height=380)
    .interactive()
)
st.altair_chart(scatter, use_container_width=True)

# 3. Rating distribution by position
st.subheader("Rating Distribution by Position")
boxplot = (
    alt.Chart(df)
    .mark_boxplot(extent="min-max")
    .encode(
        x=alt.X("position:N", title="Player Position"),
        y=alt.Y("overall_rating:Q", title="Overall Rating", scale=alt.Scale(zero=False)),
        color=alt.Color("position:N", legend=None),
        tooltip=["position", "overall_rating"],
    )
    .properties(title="Rating Distribution by Position", width="container", height=360)
    .interactive()
)
st.altair_chart(boxplot, use_container_width=True)

# 4. Market Value by Position (strip + median)
st.subheader("Market Value by Position")
strip = (
    alt.Chart(df)
    .mark_tick(thickness=1.5, size=30, opacity=0.4)
    .encode(
        x=alt.X("position:N", title="Position"),
        y=alt.Y("market_value_million_eur:Q", title="Market Value (€)", scale=alt.Scale(zero=False)),
        color=alt.Color("position:N", legend=None),
        tooltip=["position", alt.Tooltip("market_value_million_eur:Q", format=",.0f")],
    )
)
median_line = (
    alt.Chart(df)
    .mark_tick(color="black", thickness=3, size=30)
    .encode(
        x="position:N",
        y=alt.Y("median(market_value_million_eur):Q", title="Market Value (€)"),
    )
)
st.altair_chart(
    (strip + median_line)
    .properties(title="Market Value Distribution by Position (black = median)", width="container", height=360)
    .interactive(),
    use_container_width=True,
)

# 5. Potential vs Overall (gap → headroom)
st.subheader("Development Headroom: Potential vs Overall Rating")
df_copy = df.copy()
df_copy["headroom"] = df_copy["potential_rating"] - df_copy["overall_rating"]
headroom = (
    alt.Chart(df_copy)
    .mark_circle(opacity=0.6)
    .encode(
        x=alt.X("overall_rating:Q", title="Current Rating", scale=alt.Scale(zero=False)),
        y=alt.Y("potential_rating:Q", title="Potential Rating", scale=alt.Scale(zero=False)),
        color=alt.Color("headroom:Q", title="Headroom",
                        scale=alt.Scale(scheme="blues", reverse=False)),
        size=alt.Size("market_value_million_eur:Q", legend=None, scale=alt.Scale(range=[20, 300])),
        tooltip=["overall_rating", "potential_rating", "headroom", "position",
                 alt.Tooltip("market_value_million_eur:Q", format=",.0f")],
    )
    .properties(title="Potential vs Overall (size = market value)", width="container", height=380)
    .interactive()
)
diag = (
    alt.Chart(pd.DataFrame({"x": [50, 100], "y": [50, 100]}))
    .mark_line(strokeDash=[4, 2], color="grey", opacity=0.5)
    .encode(x="x:Q", y="y:Q")
)
st.altair_chart(headroom + diag, use_container_width=True)

# 6. Goals & Assists by Position
st.subheader("Goals & Assists by Position")
metric = st.radio("Metric", ["goals", "assists"], horizontal=True)
bar = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x=alt.X("position:N", title="Position"),
        y=alt.Y(f"mean({metric}):Q", title=f"Avg {metric.capitalize()}"),
        color=alt.Color("position:N", legend=None),
        tooltip=["position", alt.Tooltip(f"mean({metric}):Q", format=".2f")],
    )
    .properties(title=f"Average {metric.capitalize()} by Position", width="container", height=320)
    .interactive()
)
st.altair_chart(bar, use_container_width=True)

# 7. Correlation heatmap
st.subheader("Correlation Heatmap")
with st.spinner("Computing correlation…"):
    cols = df.select_dtypes(include="number").columns.tolist()
    corr = df[cols].corr().reset_index().melt(id_vars="index")
    corr.columns = ["Variable 1", "Variable 2", "Correlation"]

    heatmap = (
        alt.Chart(corr)
        .mark_rect()
        .encode(
            x=alt.X("Variable 1:N", title=None),
            y=alt.Y("Variable 2:N", title=None),
            color=alt.Color(
                "Correlation:Q",
                scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                title="Pearson r",
            ),
            tooltip=[
                alt.Tooltip("Variable 1:N"),
                alt.Tooltip("Variable 2:N"),
                alt.Tooltip("Correlation:Q", format=".2f"),
            ],
        )
        .properties(title="Correlation Matrix of Numerical Features", width="container", height=520)
    )
    text = heatmap.mark_text(fontSize=9).encode(
        text=alt.Text("Correlation:Q", format=".2f"),
        color=alt.condition(
            alt.expr.abs(alt.datum.Correlation) > 0.5, alt.value("white"), alt.value("black")
        ),
    )
    st.altair_chart(heatmap + text, use_container_width=True)

# 8. Market value vs minutes played
st.subheader("Market Value vs Minutes Played")
mv_chart = (
    alt.Chart(df)
    .mark_circle(opacity=0.55, size=50)
    .encode(
        x=alt.X("minutes_played:Q", title="Minutes Played", scale=alt.Scale(zero=False)),
        y=alt.Y("market_value_million_eur:Q", title="Market Value (€)", scale=alt.Scale(zero=False)),
        color=alt.Color("position:N", title="Position"),
        tooltip=["position", "minutes_played",
                 alt.Tooltip("market_value_million_eur:Q", format=",.0f"),
                 "overall_rating"],
    )
    .properties(title="Market Value vs Minutes Played", width="container", height=360)
    .interactive()
)
st.altair_chart(mv_chart, use_container_width=True)

# 9. Injury prone breakdown
st.subheader("Injury Prone Breakdown")
inj = (
    df.groupby(["position", "injury_prone"])
    .size()
    .reset_index(name="count")
)
inj_chart = (
    alt.Chart(inj)
    .mark_bar()
    .encode(
        x=alt.X("position:N", title="Position"),
        y=alt.Y("count:Q", title="Players"),
        color=alt.Color(
            "injury_prone:N",
            scale=alt.Scale(domain=["No", "Yes"], range=["#4C78A8", "#E45756"]),
            title="Injury Prone",
        ),
        tooltip=["position", "injury_prone", "count"],
    )
    .properties(title="Injury Prone Players by Position", width="container", height=300)
    .interactive()
)
st.altair_chart(inj_chart, use_container_width=True)