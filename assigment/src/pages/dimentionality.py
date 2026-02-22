import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pages.data_preparation import df_scaled_encoded

df = df_scaled_encoded.copy()

technique = "PCA"

with st.sidebar:
    technique = st.selectbox("Select Reduction Technique", ("PCA", "TSNE"))

st.header("Dimentionalty Reduction")
st.subheader(f"Using {technique}:")

st.space()

if technique == "PCA":
    pca = PCA()
    pca_2d  = PCA(n_components=2)

    pca.fit(df)

    explained_var = pca.explained_variance_ratio_
    cum_explained_var = np.cumsum(explained_var)

    pca_df = pd.DataFrame(
        data={
            'PC': range(1, len(cum_explained_var) + 1),
            'EV' : explained_var,
            'CEV': cum_explained_var
        }
    )

    bars = alt.Chart(pca_df).mark_bar(opacity=0.6).encode(
        x=alt.X("PC:O", title="Number of Components"),
        y=alt.Y("CEV:Q", title="Cumulative Explained Variance"),
        tooltip=['PC', 'CEV']
    )

    line = alt.Chart(pca_df).mark_line(color='red', point=True).encode(
        x=alt.X("PC:O"),
        y=alt.Y("CEV:Q"),
        tooltip=['PC', 'CEV']
    )

    chart = (bars + line).properties(
        title="Explained Variance by Component",
        width="container",
        height=400
    )

    st.altair_chart(chart, width='stretch')

    pca_2d_df = pd.DataFrame(pca_2d.fit_transform(df), columns=['PC1', 'PC2'])

    chart = alt.Chart(pca_2d_df).mark_circle(size=60).encode(
        x=alt.X('PC1', scale=alt.Scale(zero=False)),
        y=alt.Y('PC2', scale=alt.Scale(zero=False)),
        tooltip=['PC1', 'PC2'],
    ).properties(
        title="N=2",
        width="container"
    )

    st.altair_chart(chart, width='stretch')
    
    with st.expander(label="What does this mean?"):
        st.markdown("""
            Each principal component is explaining roughly the same amount of variance.
                    
            - The features are not strongly correlated
            - There are no dominant direction of variance
            - Information is spread evenly across features
                    
            This suggests The features are not strongly linearly dependent
                    
            e.g
                    
            - A young player can have high value
            - An older player can also have high value
            - High minutes doesn’t always mean high goals
                    
            ## Conculsion
                PCA is unnecessary for dimensionality reduction.   
        """)

else:
    loaded = False
    with st.spinner():
        perplexity = st.slider("Perplexity:", value=150, min_value=10, max_value=150, step=10)

        tsne = TSNE(n_components=2, perplexity=perplexity)

        tsne_reduced_data = tsne.fit_transform(df)

        tsne_2d_df = pd.DataFrame(tsne_reduced_data, columns=["Feature 1", "Feature 2"])

        chart = alt.Chart(tsne_2d_df).mark_circle(size=60).encode(
            x=alt.X('Feature 1', scale=alt.Scale(zero=False)),
            y=alt.Y('Feature 2', scale=alt.Scale(zero=False)),
            tooltip=['Feature 1', 'Feature 2'],
        ).properties(
            width="container"
        )

        loaded = True

    if loaded:
        st.altair_chart(chart, width='stretch')


