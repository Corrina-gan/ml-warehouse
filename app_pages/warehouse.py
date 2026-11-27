import streamlit as st

from app_pages.warehouse_view import show_warehouse

st.caption(
    "Both robots on the rware-tiny-2ag-v2 grid, for one 500-step episode. A robot scores by "
    "carrying a requested shelf onto a drop-off tile. It then has to put that shelf back in "
    "storage before it can fetch the next one. The colours are explained under the grid."
)

show_warehouse()

