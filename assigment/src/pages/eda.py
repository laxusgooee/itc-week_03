import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
from pages.data_summary import df_no_index as df


st.title("EDA")


with st.container():
    chart = alt.Chart(df).mark_bar().encode(
        alt.X("age", bin=alt.Bin(maxbins=15), title="Age"),
        alt.Y("count()", title="Count")
    ).properties(
        title="Age Distribution of Professional Football Players",
        width="container"
    ).interactive()

    st.altair_chart(chart, width='stretch')

with st.container():
    selection = alt.selection_point(fields=['position'], bind='legend')

    scatter = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X('age:Q', title='Age', scale=alt.Scale(zero=False)),
        y=alt.Y('overall_rating:Q', title='Overall Rating', scale=alt.Scale(zero=False)),
        color=alt.Color('position:N', title='Position'),
        tooltip=['age', 'overall_rating', 'position'],
        opacity=alt.condition(selection, alt.value(1), alt.value(0.1))
    ).add_params(
        selection
    ).properties(
        title='Impact of Age on Overall Rating Across Positions',
        width='container',
        height=500
    ).interactive()

    st.altair_chart(scatter, width='stretch')

with st.container():
    boxplot = alt.Chart(df).mark_boxplot(extent='min-max').encode(
        x=alt.X('position:N', title='Player Position'),
        y=alt.Y('overall_rating:Q', 
                title='Overall Rating', 
                scale=alt.Scale(zero=False)
            ),
        color=alt.Color('position:N', legend=None),
        tooltip=['position', 'overall_rating']
    ).properties(
        title='Rating Distribution by Position',
        width='container',
        height=400
    ).interactive()

    st.altair_chart(boxplot, width='stretch')

with st.container():
    with st.spinner("Wait for it...", show_time=True):
        cols = df.head().select_dtypes(include='number').columns.tolist()
        corr_matrix = df[cols].corr().reset_index().melt(id_vars='index')
        corr_matrix.columns = ['Variable 1', 'Variable 2', 'Correlation']

        heatmap = alt.Chart(corr_matrix).mark_rect().encode(
            x=alt.X('Variable 1:N', title=None),
            y=alt.Y('Variable 2:N', title=None),
            color=alt.Color('Correlation:Q', 
                            scale=alt.Scale(scheme='redblue', domain=[-1, 1]),
                            title="Pearson Corr"),
            tooltip=[
                alt.Tooltip('Variable 1:N'),
                alt.Tooltip('Variable 2:N'),
                alt.Tooltip('Correlation:Q', format='.2f')
            ]
        ).properties(
            title='Correlation Matrix of Key Metrics',
            width=600,
            height=600
        )

        text = heatmap.mark_text().encode(
            text=alt.Text('Correlation:Q', format='.2f'),
            color=alt.condition(
                abs(alt.datum.Correlation) > 0.5,
                alt.value('white'),
                alt.value('black')
            )
        )

    st.altair_chart(heatmap + text, width='stretch')