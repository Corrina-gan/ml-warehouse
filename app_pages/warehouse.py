import streamlit as st

from app_pages.shelf_request import show_shelf_request
from app_pages.warehouse_view import show_warehouse

st.caption(
    "Random plays the warehouse's own request queue, before training and after "
    "training, on the same episode. Purple shelves are in storage, teal shelves "
    "are requested, and the dark tiles are the delivery goals. Shelf ID asks for "
    "one shelf on this grid, such as S001, and plays a saved policy."
)

request_mode = st.segmented_control(
    "Request mode",
    ["Random", "Shelf ID"],
    default="Random",
    key="warehouse_request_mode",
    width="stretch",
)
if request_mode == "Shelf ID":
    show_shelf_request()
else:
    show_warehouse()

