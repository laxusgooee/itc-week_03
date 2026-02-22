import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pages.data_summary import df_no_index
from pages.data_preparation import df_scaled_encoded

df = pd.DataFrame(df_scaled_encoded)
k_means_df = df.copy()

clusters = range(2, len( df.columns.tolist()))

def plot_elbow(df, y_title="Score"):
    line = alt.Chart(df).mark_line(color="crimson").encode(
        x=alt.X("k:O", title="Number of clusters", scale=alt.Scale(zero=False)),
        y=alt.Y("score:Q", title=y_title, scale=alt.Scale(zero=False)),
    ).properties()

    dots = alt.Chart(df).mark_circle(size=120).encode(
        x=alt.X("k:O", scale=alt.Scale(zero=False)),
        y=alt.Y("score:Q", scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("k:O", title="K (No. of Clusters)"),
            alt.Tooltip("score:Q", title=y_title)
        ]
    ).properties(
        width='container',
        height=400
    )

    st.altair_chart(line + dots)

with st.container():
    selection = "WCSS"

    col_1, col_2 = st.columns([0.7, 0.3], vertical_alignment="bottom")

    with col_1:
        st.subheader("Selecting K with the elbow method")

    with col_2:
        selection = st.segmented_control(
            "Method", ("WCSS", "silhouette Score"), selection_mode="single", default=selection
        )
        
    st.space()
    
    if selection == "WCSS":
        WCSS = []
        for k in clusters:
            model = KMeans(n_clusters=k, random_state=42)
            
            model.fit(df)
            prediction = model.predict(k_means_df)

            wcss = model.inertia_

            WCSS.append((k,wcss))

        plot_elbow(pd.DataFrame(WCSS, columns=["k", "score"]), y_title="WCSS")

    else:
        sil_score = []
        for k in clusters:
            model = KMeans(n_clusters=k, random_state=42)
            preds  = model.fit_predict((df))
            score =  silhouette_score(k_means_df, preds)
            sil_score.append((k, score))

        plot_elbow(pd.DataFrame(sil_score, columns=["k", "score"]), y_title="Sil score")

    

    kmeans = KMeans(n_clusters=9)
    kmeans.fit(k_means_df)
    
    with st.expander("Observation"):
        st.markdown("""
            Our Silhouette scoreis highest at `k=9`
            WCSS elbow appears around `k=9`
                    
            this means statiscally, **9** is our optimal number of clusters.
        """)

        st.space()
        tsne = TSNE(n_components=2)

        tsne_reduced_data = tsne.fit_transform(k_means_df)

        tsne_2d_df = pd.DataFrame(tsne_reduced_data, columns=["Feature 1", "Feature 2"])

        tsne_2d_df['k_means_segments'] = kmeans.labels_

        chart = alt.Chart(tsne_2d_df).mark_circle(size=60).encode(
            x=alt.X("Feature 1", scale=alt.Scale(zero=False)),
            y=alt.Y("Feature 2", scale=alt.Scale(zero=False)),
            color=alt.Color("k_means_segments", legend=None)
        ).properties(
            width="container",
            title="TSNE Validation with K-means"
        )

        st.altair_chart(chart, width='stretch')

        st.space()

    with st.expander("Summary"):
        
        df_ori = df_no_index.copy()
        df_ori['K_means_segments'] = kmeans.labels_
        km_cluster_profile  = df_ori.groupby('K_means_segments').mean(numeric_only=True)

        st.write(km_cluster_profile.style.highlight_max(color='magenta', axis=0))
