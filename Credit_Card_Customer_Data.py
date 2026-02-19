import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

all_techniques = ("PCA", "TSNE")

st.title("Dimentionality Reduction")

# read file
df = pd.read_csv("Credit_Card_Customer_Data.csv")

# clean
df.drop(['Sl_No'], axis=1, inplace=True)
df.drop(['Customer Key'], axis=1, inplace=True)

with st.container():
    col_1, col_2, col_3 = st.columns(3, border=True)

    with col_1:
        st.subheader("Count:")
        st.header(df['Avg_Credit_Limit'].count())

    with col_2:
        st.subheader("Mean:")
        st.header(round(df['Avg_Credit_Limit'].mean(), 2))

    with col_3:
        st.subheader("Median:")
        st.header(round(df['Avg_Credit_Limit'].median(), 2))

st.subheader("Data Summary")
st.write(df.head())

col_1, *col_2, col_3 = st.columns(3, vertical_alignment="bottom", gap="large")
with col_1:
    st.subheader("")

with col_3:
    scalled = st.toggle("Use Scalled Data")

with st.container(border=True):
    techniques = st.multiselect("Techniques", all_techniques, default=all_techniques[0])
    

frame = df

if scalled:
    scaler = StandardScaler()
    transformer = PowerTransformer(method='yeo-johnson')

    df[['Avg_Credit_Limit']]  = transformer.fit_transform(df[['Avg_Credit_Limit']])
    frame = scaler.fit_transform(df)

data = df['Avg_Credit_Limit']

fig, ax = plt.subplots(1,2, figsize=(10,4))
sns.boxplot(data, ax=ax[0])
sns.histplot(data, kde=True, ax=ax[1])

plt.tight_layout()

st.pyplot(fig)


st.space()


def show_pca(df):
    pca = PCA()

    pca.fit(df)

    explained_var = pca.explained_variance_ratio_
    cum_explained_var = np.cumsum(explained_var)

    fig = plt.figure()
    plt.plot(range(1, len(cum_explained_var) + 1), cum_explained_var, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    st.pyplot(fig)

    pca_2d  = PCA(n_components=2)

    pca_2d_data =  pca_2d.fit_transform(df)
    pca_2d_df = pd.DataFrame(pca_2d_data, columns=['PC1', 'PC2'])

    fig, ax = plt.subplots()
    sns.scatterplot(data=pca_2d_df, x="PC1", y="PC2", ax=ax)
    st.pyplot(fig)

   
def show_tsne(df, perplexity=0.0):
    tsne = TSNE(n_components=2, perplexity=perplexity)

    tsne_reduced_data = tsne.fit_transform(df)

    tsne_2d_data = pd.DataFrame(tsne_reduced_data, columns=["Feature 1", "Feature 2"])

    fig, ax = plt.subplots()
    sns.scatterplot(data=tsne_2d_data, x= "Feature 1", y = "Feature 2", ax=ax)
    st.pyplot(fig)


if 'PCA' in techniques:
    with st.expander(label="PCA"):
        show_pca(frame)

if 'TSNE' in techniques:
    with st.expander(label="TSNE"):
        perplexity = st.slider("Select Perpexity", value=50.0, step=10.0, min_value=10.0, max_value=150.00)
        show_tsne(frame, perplexity=perplexity)

