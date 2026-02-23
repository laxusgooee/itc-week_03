import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pages.clustering.data_preparation import df_scaled_encoded

df = df_scaled_encoded.copy()

technique = "PCA"

with st.sidebar:
    technique = st.selectbox("Select Reduction Technique", ("PCA", "TSNE"))

st.header("Dimensionality Reduction")
st.subheader(f"Using {technique}:")

st.divider()

if technique == "PCA":
    pca = PCA()
    pca_2d = PCA(n_components=2, random_state=42)

    pca.fit(df)

    explained_var = pca.explained_variance_ratio_
    cum_explained_var = np.cumsum(explained_var)

    # How many components to reach 90% variance
    n_90 = int(np.searchsorted(cum_explained_var, 0.90)) + 1

    pca_df = pd.DataFrame(
        data={
            "PC": range(1, len(cum_explained_var) + 1),
            "EV": explained_var,
            "CEV": cum_explained_var,
        }
    )

    # Individual variance bars
    bars = (
        alt.Chart(pca_df)
        .mark_bar(opacity=0.6, color="#4C78A8")
        .encode(
            x=alt.X("PC:O", title="Principal Component"),
            y=alt.Y("EV:Q", title="Explained Variance Ratio", axis=alt.Axis(format=".0%")),
            tooltip=[
                alt.Tooltip("PC:O", title="Component"),
                alt.Tooltip("EV:Q", title="Explained Variance", format=".2%"),
                alt.Tooltip("CEV:Q", title="Cumulative Variance", format=".2%"),
            ],
        )
    )

    # Cumulative variance line
    line = (
        alt.Chart(pca_df)
        .mark_line(color="crimson", point=True)
        .encode(
            x=alt.X("PC:O"),
            y=alt.Y("CEV:Q"),
            tooltip=[
                alt.Tooltip("PC:O", title="Component"),
                alt.Tooltip("CEV:Q", title="Cumulative Variance", format=".2%"),
            ],
        )
    )

    # 90% threshold reference line
    threshold_df = pd.DataFrame({"y": [0.90]})
    threshold_line = (
        alt.Chart(threshold_df)
        .mark_rule(color="orange", strokeDash=[6, 4])
        .encode(y=alt.Y("y:Q"))
    )

    chart = (bars + line + threshold_line).properties(
        title=f"Scree Plot — {n_90} components explain 90% of variance",
        width="container",
        height=400,
    )

    st.altair_chart(chart, use_container_width=True)

    st.info(f"**{n_90} principal components** are needed to explain **90%** of the total variance.")

    # 2D PCA scatter
    pca_2d_df = pd.DataFrame(pca_2d.fit_transform(df), columns=["PC1", "PC2"])

    scatter = (
        alt.Chart(pca_2d_df)
        .mark_circle(size=60, opacity=0.6)
        .encode(
            x=alt.X("PC1:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("PC2:Q", scale=alt.Scale(zero=False)),
            tooltip=["PC1:Q", "PC2:Q"],
        )
        .properties(
            title=f"PCA Projection (n=2) — {explained_var[:2].sum():.1%} of variance captured",
            width="container",
            height=400,
        )
        .interactive()
    )

    st.altair_chart(scatter, use_container_width=True)

    with st.expander(label="What does this mean?"):
        st.markdown(f"""
Each principal component explains roughly the same amount of variance (no single dominant direction).

- **{n_90} components** are needed to explain 90% of the data's variance
- The features are **not strongly linearly correlated**
- Information is spread evenly across all features

This is typical for football player data where:
- A young player can have high market value
- An older player can also have high market value
- High minutes doesn't always mean high goals

## Conclusion
PCA alone is **not ideal** for aggressive dimensionality reduction here. t-SNE is better for cluster visualisation.
        """)

else:
    # Move slider above the spinner so UI is responsive
    perplexity = st.slider("Perplexity:", value=30, min_value=5, max_value=150, step=5)

    with st.spinner("Running t-SNE… this may take a moment"):
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_reduced_data = tsne.fit_transform(df)
        tsne_2d_df = pd.DataFrame(tsne_reduced_data, columns=["Feature 1", "Feature 2"])

        chart = (
            alt.Chart(tsne_2d_df)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X("Feature 1:Q", scale=alt.Scale(zero=False)),
                y=alt.Y("Feature 2:Q", scale=alt.Scale(zero=False)),
                tooltip=["Feature 1:Q", "Feature 2:Q"],
            )
            .properties(
                title=f"t-SNE Projection (perplexity={perplexity})",
                width="container",
                height=450,
            )
            .interactive()
        )

        st.altair_chart(chart, use_container_width=True)

    with st.expander("What does perplexity control?"):
        st.markdown("""
**Perplexity** roughly controls the effective number of neighbours each point considers.

| Value | Effect |
|-------|--------|
| Low (5–15) | Emphasises tight local clusters |
| Medium (30–50) | Balanced — usually best starting point |
| High (100+) | More global structure, clusters may merge |

Try **30–50** first, then adjust based on cluster separation.
        """)
