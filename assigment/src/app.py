import streamlit as st


def get_page_path(page):
    return f"pages/{page}.py"

pages = {
    "Navigation": [
        st.Page("pages/intro.py", title="Welcome", default=True),
        st.Page("pages/data_summary.py", title="Preprocessing"),
        st.Page("pages/eda.py", title="EDA"),
        st.Page("pages/data_preparation.py", title="Data Preparation"),
        st.Page("pages/dimentionality.py", title="Dimentionality"),
        st.Page(get_page_path("clustering"), title="Clustering"),
        st.Page("pages/extro.py", title="Business Insights"),
    ],
}


pg = st.navigation(pages)
pg.run()