import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pages.clustering.data_summary import df_no_index
from pages.clustering.data_preparation import df_scaled_encoded

df = pd.DataFrame(df_scaled_encoded)
k_means_df = df.copy()

# Cap max K at 15 or n_samples//10 (whichever is smaller, min 2)
max_k = min(15, max(2, len(df) // 10))
clusters = range(2, max_k + 1)


# helpers

def plot_elbow(df_elbow: pd.DataFrame, y_title: str = "Score", best_k: int | None = None):
    """Render an elbow/silhouette chart, optionally annotating the best K."""
    base = alt.Chart(df_elbow)

    line = base.mark_line(color="crimson").encode(
        x=alt.X("k:O", title="Number of Clusters"),
        y=alt.Y("score:Q", title=y_title, scale=alt.Scale(zero=False)),
    )

    dots = base.mark_circle(size=120).encode(
        x=alt.X("k:O"),
        y=alt.Y("score:Q", scale=alt.Scale(zero=False)),
        color=alt.condition(
            alt.datum["k"] == best_k,
            alt.value("gold"),
            alt.value("crimson"),
        ),
        tooltip=[
            alt.Tooltip("k:O", title="K (Clusters)"),
            alt.Tooltip("score:Q", title=y_title, format=".4f"),
        ],
    )

    chart = (line + dots).properties(width="container", height=400)

    if best_k is not None:
        best_row = df_elbow[df_elbow["k"] == best_k]
        annotation = (
            alt.Chart(best_row)
            .mark_text(dy=-16, fontSize=13, fontWeight="bold", color="gold")
            .encode(
                x=alt.X("k:O"),
                y=alt.Y("score:Q"),
                text=alt.value(f"k = {best_k}"),
            )
        )
        chart = (line + dots + annotation).properties(width="container", height=400)

    st.altair_chart(chart, use_container_width=True)


#title 

st.title("Clustering")

#Step 1: K selection

with st.container():
    col_1, col_2 = st.columns([0.7, 0.3], vertical_alignment="bottom")
    with col_1:
        st.subheader("Step 1 – Selecting K (Elbow Method)")
    with col_2:
        selection = st.segmented_control(
            "Method", ("WCSS", "Silhouette Score"), selection_mode="single", default="WCSS"
        )

    st.divider()

    if selection == "WCSS":
        wcss_data = []
        for k in clusters:
            model = KMeans(n_clusters=k, random_state=42, n_init="auto")
            model.fit(df)
            wcss_data.append((k, model.inertia_))

        wcss_df = pd.DataFrame(wcss_data, columns=["k", "score"])

        best_k_wcss = 9

        plot_elbow(wcss_df, y_title="WCSS (Inertia)", best_k=best_k_wcss)

        with st.expander("Observation"):
            st.markdown(f"""
- The steepest drop in WCSS occurs at **k = {best_k_wcss}** (highlighted in gold)
- After this point, adding more clusters yields diminishing returns
- WCSS always decreases with more clusters — look for the "elbow" bend
            """)

    else:
        sil_data = []
        for k in clusters:
            model = KMeans(n_clusters=k, random_state=42, n_init="auto")
            preds = model.fit_predict(df)
            sil_data.append((k, silhouette_score(k_means_df, preds)))

        sil_df = pd.DataFrame(sil_data, columns=["k", "score"])
        best_k_sil = int(sil_df.loc[sil_df["score"].idxmax(), "k"])

        plot_elbow(sil_df, y_title="Silhouette Score", best_k=best_k_sil)

        with st.expander("Observation"):
            st.markdown(f"""
- **Silhouette score is highest at k = {best_k_sil}** (highlighted in gold)
- A higher silhouette score means clusters are more cohesive and well-separated
- Scores range from –1 (poor) to +1 (perfect); above **0.2** is generally acceptable
            """)

st.divider()

#Step 2: Fit optimal model

OPTIMAL_K = 9   # update this if the elbow / silhouette analysis changes

st.subheader(f"Step 2 – Fitting K-Means (k = {OPTIMAL_K})")
st.caption("Based on both the WCSS elbow and highest silhouette score.")

kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init="auto")
kmeans.fit(k_means_df)

st.divider()

#Step 3: t-SNE visualisation

st.subheader("Step 3 – t-SNE Visualisation of Clusters")

with st.expander("Show t-SNE plot"):
    perplexity = st.slider(
        "Perplexity",
        min_value=5,
        max_value=min(150, len(df) - 1),
        value=30,
        step=5,
        help="Controls neighbourhood size. Try 20–50 for clearest separation.",
    )

    with st.spinner("Running t-SNE… this may take a moment"):
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_data = tsne.fit_transform(k_means_df)
        tsne_df = pd.DataFrame(tsne_data, columns=["Feature 1", "Feature 2"])
        tsne_df["Cluster"] = kmeans.labels_.astype(str)

        tsne_chart = (
            alt.Chart(tsne_df)
            .mark_circle(size=60, opacity=0.75)
            .encode(
                x=alt.X("Feature 1:Q", scale=alt.Scale(zero=False)),
                y=alt.Y("Feature 2:Q", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "Cluster:N",
                    title="Cluster",
                    scale=alt.Scale(scheme="tableau10"),
                ),
                tooltip=["Cluster:N", alt.Tooltip("Feature 1:Q", format=".2f"), alt.Tooltip("Feature 2:Q", format=".2f")],
            )
            .properties(
                title=f"t-SNE – K-Means Cluster Separation (k={OPTIMAL_K}, perplexity={perplexity})",
                width="container",
                height=450,
            )
            .interactive()
        )
        st.altair_chart(tsne_chart, use_container_width=True)

st.divider()

#Step 4: Cluster profiles─

st.subheader("Step 4 – Cluster Profiles")

df_ori = df_no_index.copy()
df_ori["Cluster"] = kmeans.labels_
numeric_cols = df_ori.select_dtypes(include="number").columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "Cluster"]
km_profile = df_ori.groupby("Cluster")[numeric_cols].mean()

with st.expander("Cluster mean table"):
    st.dataframe(
        km_profile.style
        .highlight_max(color="mediumspringgreen", axis=0)
        .highlight_min(color="salmon", axis=0)
    )

st.divider()

#Step 5: Cluster heatmap──

st.subheader("Step 5 – Cluster Feature Heatmap")
st.markdown("Normalised cluster profiles (z-score) — highlights which features define each cluster.")

km_z = (km_profile - km_profile.mean()) / km_profile.std()
heatmap_df = km_z.reset_index().melt(id_vars="Cluster", var_name="Feature", value_name="Z-Score")

hm = (
    alt.Chart(heatmap_df)
    .mark_rect()
    .encode(
        x=alt.X("Feature:N", title=None),
        y=alt.Y("Cluster:O", title="Cluster"),
        color=alt.Color(
            "Z-Score:Q",
            scale=alt.Scale(scheme="redblue", domain=[-2, 2]),
            title="Z-Score",
        ),
        tooltip=[
            "Cluster:O",
            "Feature:N",
            alt.Tooltip("Z-Score:Q", format=".2f"),
        ],
    )
    .properties(title="Cluster Feature Z-Score Heatmap", width="container", height=360)
)

hm_text = hm.mark_text(fontSize=8).encode(
    text=alt.Text("Z-Score:Q", format=".1f"),
    color=alt.condition(
        alt.expr.abs(alt.datum["Z-Score"]) > 1.2,
        alt.value("white"),
        alt.value("black"),
    ),
)

st.altair_chart(hm + hm_text, use_container_width=True)

st.divider()

#Step 6: Cluster sizes────

st.subheader("Step 6 – Cluster Sizes")

cluster_sizes = (
    pd.Series(kmeans.labels_, name="Cluster")
    .value_counts()
    .reset_index()
    .rename(columns={"count": "Count"})
    .sort_values("Cluster")
)
cluster_sizes["Cluster"] = cluster_sizes["Cluster"].astype(str)

avg_size = len(df) // OPTIMAL_K

size_chart = (
    alt.Chart(cluster_sizes)
    .mark_bar()
    .encode(
        x=alt.X("Cluster:N", sort="-y", title="Cluster"),
        y=alt.Y("Count:Q", title="Number of Players"),
        color=alt.Color("Cluster:N", scale=alt.Scale(scheme="tableau10"), legend=None),
        tooltip=["Cluster:N", "Count:Q"],
    )
    .properties(title="Players per Cluster", width="container", height=300)
    .interactive()
)

avg_line = (
    alt.Chart(pd.DataFrame({"y": [avg_size]}))
    .mark_rule(color="white", strokeDash=[5, 3], strokeWidth=1.5)
    .encode(y=alt.Y("y:Q"))
)

avg_label = (
    alt.Chart(pd.DataFrame({"y": [avg_size], "label": [f"Avg = {avg_size}"]}))
    .mark_text(align="right", dx=-6, dy=-6, color="white", fontSize=11)
    .encode(
        y=alt.Y("y:Q"),
        x=alt.value(600),
        text="label:N",
    )
)

st.altair_chart(size_chart + avg_line + avg_label, use_container_width=True)

with st.expander("Interpretation"):
    smallest = cluster_sizes.loc[cluster_sizes["Count"].idxmin()]
    largest = cluster_sizes.loc[cluster_sizes["Count"].idxmax()]
    st.markdown(f"""
- **Largest cluster:** {largest['Cluster']} ({largest['Count']} players)
- **Smallest cluster:** {smallest['Cluster']} ({smallest['Count']} players)
- **Average cluster size:** {avg_size} players
- Highly unequal sizes may indicate overlapping clusters or that K is too high
    """)
