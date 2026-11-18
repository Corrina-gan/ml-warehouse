import streamlit as st

from app_pages.warehouse_view import show_warehouse

st.caption(
    "Both robots on the rware-tiny-2ag-v2 grid. Purple shelves are in storage, "
    "teal shelves are requested, and the dark tiles are the delivery goals. "
    "Light gray cells are the aisles."
)

show_warehouse()

