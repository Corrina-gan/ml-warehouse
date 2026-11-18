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
            title="Analysis",
            icon=":material/query_stats:",
            default=True,
        ),
        st.Page(
            "app_pages/warehouse.py",
            title="Warehouse",
            icon=":material/warehouse:",
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
