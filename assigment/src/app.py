import streamlit as st

def get_page_path(module, page):
    return f"pages/{module}/{page}.py"

pages = {
    "Clustering": [
        st.Page(get_page_path("clustering","intro"), title="Welcome", default=True),
        st.Page(get_page_path("clustering","data_summary"), title="Preprocessing"),
        st.Page(get_page_path("clustering","eda"), title="EDA"),
        st.Page(get_page_path("clustering","data_preparation"), title="Data Preparation"),
        st.Page(get_page_path("clustering","dimentionality"), title="Dimentionality"),
        st.Page(get_page_path("clustering","clustering"), title="Clustering"),
        st.Page(get_page_path("clustering","extro"), title="Business Insights"),
    ],
    "Time Series": [
        st.Page(get_page_path("time_series","t_intro"), title="Welcome"),
        st.Page(get_page_path("time_series","t_data_summary"), title="Preprocessing"),
        st.Page(get_page_path("time_series","t_data_preparation"), title="Data Preparation"),
        st.Page(get_page_path("time_series","t_eda"), title="EDA"),
        st.Page(get_page_path("time_series","t_hypothesis"), title="Hypothesis Testing"),
        st.Page(get_page_path("time_series","t_arima"), title="ARIMA"),
        st.Page(get_page_path("time_series","t_sarima"), title="SARIMA"),
        st.Page(get_page_path("time_series","t_sarimax"), title="SARIMAX"),
        st.Page(get_page_path("time_series","t_compare"), title="Model Comparison"),
        st.Page(get_page_path("time_series","t_extro"), title="Business Insights"),
    ]
}

pg = st.navigation(pages)
pg.run()