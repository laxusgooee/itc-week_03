import streamlit as st
from scripts.data import df

df_no_index = df.drop(['player_id', 'player_name'], axis=1)

@st.dialog("Data types")
def show_types():
    st.write(df.dtypes)

st.title("Our Data")

with st.container():
    col_1, col_2, col_3 = st.columns(3, border=True)

    with col_1:
        st.subheader("Shape:")
        st.header(df_no_index.shape)

    with col_2:
        st.subheader("Null Values:")
        st.header(all(i > 0 for i in df_no_index.isna().sum()))

    with col_3:
        st.subheader("Duplicates:")
        st.header(df_no_index.duplicated().sum())

with st.container():
    show_indexs = True
    col_1, col_2 = st.columns([0.7, 0.3], vertical_alignment='center')

    with col_1:
        st.subheader("Data Summary")
    with col_2:
        with st.container(height="content", vertical_alignment="bottom" , horizontal_alignment="right"):
            show_indexs = st.toggle('Hide Indexes', value=show_indexs, help="We are hiding player_id and player_name")

    st.write(df_no_index.head() if show_indexs else df.head())

with st.container():
    col_1, col_2 = st.columns([0.6,0.4], vertical_alignment='bottom')

    with col_1:
        st.subheader("Data Description")
    with col_2:
        with st.container( horizontal_alignment='right'):
            st.button("Show types", on_click=show_types)

    st.write(df_no_index.describe())
