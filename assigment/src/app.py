import streamlit as st

pages = {
    "Navigation": [
        st.Page("pages/intro.py", title="Welcome"),
        st.Page("pages/data_summary.py", title="Preprocessing"),
        st.Page("pages/eda.py", title="EDA"),
        st.Page("pages/extro.py", title="Create your account"),
    ],
}


pg = st.navigation(pages, expanded=True)
pg.run()