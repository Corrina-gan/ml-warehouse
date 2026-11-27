import streamlit as st

st.set_page_config(
    page_title="Warehouse robots",
    page_icon=":material/warehouse:",
    layout="wide",
)

page = st.navigation(
    [
        st.Page(
            "app_pages/analysis.py",
            title="Exploratory data analysis",
            icon=":material/query_stats:",
            default=True,
        ),
        st.Page(
            "app_pages/warehouse.py",
            title="Warehouse simulation",
            icon=":material/warehouse:",
        ),
        st.Page(
            "app_pages/results.py",
            title="Results and comparison",
            icon=":material/leaderboard:",
        ),
        st.Page(
            "app_pages/selection.py",
            title="Model selection",
            icon=":material/checklist:",
        ),
        st.Page(
            "app_pages/tuning.py",
            title="Hyperparameter search",
            icon=":material/tune:",
        ),
        st.Page(
            "app_pages/shared.py",
            title="Shared setup",
            icon=":material/account_tree:",
        ),
    ],
    position="top",
)

st.title(page.title)
page.run()
