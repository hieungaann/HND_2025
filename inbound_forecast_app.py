import io
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Inbound Planner", layout="wide")

st.title("📦 Hieu Ngan's Inbound Planner")

with st.expander("ℹ️ Input data format"):
    st.markdown("""
    **Upload file CSV hoặc Excel (.xlsx)** theo cấu trúc giống file *Template Shop LG* với các cột quan trọng:
    - Cột thông tin cơ bản: `sku_id`, `mt_sku_id`, `shop_id`, `shop_name`, `item_name`, `category_cluster`
    - Tồn kho: `total_stock_vncb`, `total_stock_vnn`, `total_stock_vns`, `total_stock_vndb`
    - Inbound: `vncb_inbounding`, `vnn_inbounding`, `vns_inbounding`, `vndb_inbounding`
    - Sales 30 ngày (TB/ngày): `l30_daily_itemsold_vncb`, `l30_daily_itemsold_vnn`, `l30_daily_itemsold_vns`, `l30_daily_itemsold_vndb`
    """)

st.sidebar.header("⚙️ Parameters")
horizon_days = st.sidebar.number_input("Forecast horizon (days)", min_value=7, max_value=180, value=90, step=1)
leadtime
