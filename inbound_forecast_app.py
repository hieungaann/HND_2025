# inbound_forecast_app.py
import io
from datetime import timedelta
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Inbound Planner", layout="wide")
st.title("📦 Hieu Ngan's Inbound Planner")

# ------------------------------
# Sidebar: choose module
# ------------------------------
mode = st.sidebar.radio("Chọn chức năng", ["Inbound Planner (HND2025)", "Excel Replenishment Tool"])

# ==============================
# Module A: Inbound Planner (HND2025)
# ==============================
if mode == "Inbound Planner (HND2025)":
    st.header("📥 Inbound Planner (HND2025)")

    with st.expander("ℹ️ Input data format"):
        st.markdown("""
        **Upload file CSV hoặc Excel (.xlsx)** theo cấu trúc giống file *Template Shop LG* với các cột quan trọng:
        - Cột thông tin cơ bản: `sku_id`, `mt_sku_id`, `shop_id`, `shop_name`, `item_name`, `category_cluster`
        - Tồn kho: `total_stock_vncb`, `total_stock_vnn`, `total_stock_vns`, `total_stock_vndb`
        - Inbound: `vncb_inbounding`, `vnn_inbounding`, `vns_inbounding`, `vndb_inbounding`
        - Sales 30 ngày (TB/ngày): `l30_daily_itemsold_vncb`, `l30_daily_itemsold_vnn`, `l30_daily_itemsold_vns`, `l30_daily_itemsold_vndb`
        """)

    # Sidebar params (HND)
    st.sidebar.header("⚙️ HND Parameters")
    horizon_days = st.sidebar.number_input("Forecast horizon (days)", min_value=7, max_value=180, value=90, step=1, key="hnd_horizon")
    leadtime_days = st.sidebar.number_input("Leadtime (days)", min_value=0, max_value=60, value=7, step=1, key="hnd_leadtime")

    st.sidebar.markdown("**Safety stock (days)**")
    safety_elha = st.sidebar.number_input("ELHA", min_value=0, max_value=90, value=30, step=1, key="hnd_elha")
    safety_fmcg = st.sidebar.number_input("FMCG", min_value=0, max_value=90, value=21, step=1, key="hnd_fmcg")
    safety_other = st.sidebar.number_input("Others", min_value=0, max_value=90, value=14, step=1, key="hnd_other")

    st.sidebar.markdown("**Constraints (optional)**")
    pack_size = st.sidebar.number_input("Pack size (round to multiples of)", min_value=0, max_value=1000, value=0, step=1, key="hnd_pack")
    moq_units = st.sidebar.number_input("MOQ (units)", min_value=0, max_value=100000, value=0, step=1, key="hnd_moq")

    # Upload
    uploaded_hnd = st.file_uploader("Upload CSV or Excel for HND (template: Template Shop LG)", type=["csv","xlsx"], key="hnd_upload")

    sample_df = None
    if uploaded_hnd:
        try:
            if uploaded_hnd.name.lower().endswith(".csv"):
                df_hnd = pd.read_csv(uploaded_hnd)
            else:
                df_hnd = pd.read_excel(uploaded_hnd)
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")
            st.stop()
    else:
        st.info("Chưa upload file HND. Bạn có thể thử với dữ liệu mẫu (nhấn nút dưới).")
        if st.button("Dùng dữ liệu mẫu (HND)"):
            sample_df = pd.DataFrame({
                "sku_id": ["A","B","C"],
                "mt_sku_id": ["111_1", "222_2", "333_3"],
                "shop_id": [1,1,1],
                "shop_name": ["LG Official Store"]*3,
                "item_name": ["SKU A","SKU B","SKU C"],
                "category_cluster": ["ELHA","FMCG","Others"],
                "total_stock_vncb": [100,50,0],
                "total_stock_vnn": [0,0,0],
                "total_stock_vns": [0,0,0],
                "total_stock_vndb": [0,0,0],
                "vncb_inbounding": [10,0,0],
                "vnn_inbounding": [0,0,0],
                "vns_inbounding": [0,0,0],
                "vndb_inbounding": [0,0,0],
                "l30_daily_itemsold_vncb": [2,4,0.5],
                "l30_daily_itemsold_vnn": [0,0,0],
                "l30_daily_itemsold_vns": [0,0,0],
                "l30_daily_itemsold_vndb": [0,0,0],
            })
            df_hnd = sample_df

    if 'df_hnd' in locals():
        # Optional filter
        st.sidebar.markdown("---")
        mt_filter = st.sidebar.text_input("Filter by mt_sku_id (optional)", key="hnd_filter")
        if mt_filter:
            df_hnd = df_hnd[df_hnd["mt_sku_id"].astype(str).str.contains(mt_filter, na=False)]

        # Compute helpers
        df_hnd["total_stock"] = df_hnd[["total_stock_vncb","total_stock_vnn","total_stock_vns","total_stock_vndb"]].sum(axis=1, skipna=True)
        df_hnd["total_inbound"] = df_hnd[["vncb_inbounding","vnn_inbounding","vns_inbounding","vndb_inbounding"]].sum(axis=1, skipna=True)
        df_hnd["avg_sales_30d"] = df_hnd[["l30_daily_itemsold_vncb","l30_daily_itemsold_vnn","l30_daily_itemsold_vns","l30_daily_itemsold_vndb"]].sum(axis=1, skipna=True)

        def safety_days(cat):
            if cat == "ELHA": return safety_elha
            if cat == "FMCG": return safety_fmcg
            return safety_other

        df_hnd["safety_stock_days"] = df_hnd["category_cluster"].fillna("Others").apply(safety_days)
        df_hnd["safety_units"] = df_hnd["safety_stock_days"] * df_hnd["avg_sales_30d"]
        df_hnd["leadtime_units"] = leadtime_days * df_hnd["avg_sales_30d"]
        df_hnd["forecast_h_units"] = horizon_days * df_hnd["avg_sales_30d"]
        df_hnd["available_units"] = df_hnd["total_stock"] + df_hnd["total_inbound"]

        df_hnd["inbound_need_units"] = (df_hnd["forecast_h_units"] + df_hnd["safety_units"] + df_hnd["leadtime_units"] - df_hnd["available_units"]).clip(lower=0)

        # Rounding by constraints
        def round_constraints(x):
            if moq_units and x > 0:
                x = max(x, moq_units)
            if pack_size and x > 0:
                x = int(np.ceil(x / pack_size) * pack_size)
            return x

        df_hnd["IB_suggest_units"] = df_hnd["inbound_need_units"].apply(round_constraints)

        # Coverage after IB
        with np.errstate(divide='ignore', invalid='ignore'):
            df_hnd["coverage_after_IB_days"] = np.where(
                df_hnd["avg_sales_30d"]>0,
                (df_hnd["available_units"] + df_hnd["IB_suggest_units"]) / df_hnd["avg_sales_30d"],
                np.nan
            )

        show_cols = [
            "sku_id","mt_sku_id","shop_name","item_name","category_cluster",
            "avg_sales_30d","total_stock","total_inbound",
            "safety_stock_days","forecast_h_units",
            "available_units","inbound_need_units","IB_suggest_units","coverage_after_IB_days"
        ]
        result_hnd = df_hnd[show_cols].copy()

        st.success(f"Đã tính xong. {len(result_hnd)} dòng.")
        st.dataframe(result_hnd, use_container_width=True)

        # Download CSV
        csv_bytes = result_hnd.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Tải kết quả CSV", data=csv_bytes, file_name="inbound_suggestion.csv", mime="text/csv")

        # Download Excel (xlsxwriter)
        output_hnd = io.BytesIO()
        with pd.ExcelWriter(output_hnd, engine="xlsxwriter") as writer:
            result_hnd.to_excel(writer, index=False, sheet_name="Inbound Plan")
        st.download_button(
            "⬇️ Tải kết quả Excel",
            data=output_hnd.getvalue(),
            file_name="inbound_suggestion.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.caption("• Công thức: Inbound = Forecast(h) + Safety + Leadtime − (Stock + Inbound). Áp dụng MOQ/Pack-size nếu có.")
    else:
        st.stop()

# ==============================
# Module B: Excel Replenishment Tool (Replenishment Auto.xlsx)
# ==============================
elif mode == "Excel Replenishment Tool":
    st.header("📂 Excel Replenishment Tool — từ file Replenishment Auto.xlsx (sheet 'Input')")

    with st.expander("ℹ️ Logic implemented (mirror Excel)"):
        st.markdown("""
        - Đọc sheet **Input** (cột mẫu: CAT, SKU_code, Available stock, Upcoming stock, Upcoming date, Forecast OB/day, Leadtime (day), DOC).
        - **Projection**: mô phỏng tồn (end-of-day) theo ngày cho một khoảng `projection_days`.
        - **ROP date**: ngày đầu tiên tồn kho ≤ (Forecast OB/day × Leadtime).
        - **Order Qty**: = DOC × Forecast OB/day (lấy nguyên, có thể áp MOQ/pack size).
        - **Arrival date** = ROP date + Leadtime.
        - Tạo 2 sheet kết quả: `Output.current` (không có order), `Output.ordered` (có order trên arrival date).
        """)

    # Sidebar params for Excel tool
    st.sidebar.header("⚙️ Replenishment Parameters")
    projection_days = st.sidebar.number_input("Projection days", min_value=7, max_value=180, value=40, step=1, key="rep_proj")
    start_date = st.sidebar.date_input("Projection start date (default = today)", value=pd.Timestamp.today().date(), key="rep_start")
    apply_rounding = st.sidebar.checkbox("Apply MOQ / Pack-size to OrderQty", value=False, key="rep_round")
    moq = st.sidebar.number_input("MOQ (units)", min_value=0, max_value=1_000_000, value=0, step=1, key="rep_moq")
    pack_size = st.sidebar.number_input("Pack size (round to multiples of)", min_value=0, max_value=10_000, value=0, step=1, key="rep_pack")

    uploaded_rep = st.file_uploader("Upload Replenishment Excel (has sheet 'Input')", type=["xlsx","xls","csv"], key="rep_upload")

    # helper functions
    def find_col(df, keywords):
        cols = list(df.columns)
        lower = [c.lower() for c in cols]
        for k in keywords:
            for i, lc in enumerate(lower):
                if k in lc:
                    return cols[i]
        return None

    def safe_num(x):
        return pd.to_numeric(x, errors="coerce").fillna(0)

    if not uploaded_rep:
        st.info("Chưa upload file Replenishment. Bạn có thể thử với dữ liệu mẫu.")
        if st.button("Dùng dữ liệu mẫu (Replenishment)"):
            df_input = pd.DataFrame({
                "CAT": ["ELHA","ELHA","FMCG","FMCG"],
                "SKU_code": ["A001","A002","A003","A004"],
                "Available stock": [400,250,186,386],
                "Upcoming stock": [68,76,61,66],
                "Upcoming date": ["2025-09-26","2025-10-05","2025-09-28","2025-10-02"],
                "Forecast OB/day": [25,30,35,40],
                "Leadtime (day)": [6,4,5,2],
                "DOC": [20,20,15,15],
            })
        else:
            st.stop()
    else:
        # read input sheet (best-effort)
        try:
            if uploaded_rep.name.lower().endswith((".xls", ".xlsx")):
                xls = pd.ExcelFile(uploaded_rep)
                if "Input" in xls.sheet_names:
                    df_input = pd.read_excel(uploaded_rep, sheet_name="Input")
                else:
                    df_input = pd.read_excel(uploaded_rep, sheet_name=0)
            else:
                df_input = pd.read_csv(uploaded_rep)
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")
            st.stop()

    # detect columns
    col_CAT = find_col(df_input, ["cat", "category"])
    col_sku = find_col(df_input, ["sku", "sku_code", "sku_id"])
    col_available = find_col(df_input, ["available stock", "available", "total_stock", "available_stock"])
    col_upcoming_stock = find_col(df_input, ["upcoming stock", "upcoming", "incoming", "incoming_qty", "upcoming_stock"])
    col_upcoming_date = find_col(df_input, ["upcoming date", "upcoming_date", "arrival date", "incoming_date"])
    col_forecast = find_col(df_input, ["forecast ob/day", "forecast", "forecast ob", "forecast per day"])
    col_leadtime = find_col(df_input, ["leadtime", "leadtime (day)", "leadtime (days)"])
    col_doc = find_col(df_input, ["doc", "days of cover", "days_of_cover"])

    required = {"CAT": col_CAT, "SKU": col_sku, "Available": col_available, "Forecast": col_forecast, "Leadtime": col_leadtime, "DOC": col_doc}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        st.error(f"Không tìm thấy cột cần thiết trong sheet 'Input'. Thiếu: {', '.join(missing)}. Cột hiện có: {', '.join(df_input.columns.tolist())}")
        st.stop()

    # normalize & rename
    df = df_input.rename(columns={
        col_CAT: "CAT",
        col_sku: "SKU",
        col_available: "Available_stock",
        col_upcoming_stock: "Upcoming_stock" if col_upcoming_stock else None,
        col_upcoming_date: "Upcoming_date" if col_upcoming_date else None,
        col_forecast: "Forecast_per_day",
        col_leadtime: "Leadtime_days",
        col_doc: "DOC"
    })
    df = df.loc[:, ~df.columns.duplicated()]

    # numeric clean
    df["Available_stock"] = safe_num(df.get("Available_stock", 0))
    if "Upcoming_stock" in df.columns:
        df["Upcoming_stock"] = safe_num(df["Upcoming_stock"])
    else:
        df["Upcoming_stock"] = 0
    if "Upcoming_date" in df.columns:
        df["Upcoming_date"] = pd.to_datetime(df["Upcoming_date"], errors="coerce").dt.normalize()
    else:
        df["Upcoming_date"] = pd.NaT

    df["Forecast_per_day"] = safe_num(df.get("Forecast_per_day", 0))
    df["Leadtime_days"] = safe_num(df.get("Leadtime_days", 0)).astype(int)
    df["DOC"] = safe_num(df.get("DOC", 0))

    # prepare dates
    start = pd.to_datetime(start_date)
    dates = pd.date_range(start=start, periods=int(projection_days), freq="D")

    def project_inventory(available, daily_demand, incoming_events, date_index):
        inv_list = []
        inv = float(available)
        for d in date_index:
            inc = incoming_events.get(pd.to_datetime(d).normalize(), 0)
            inv += inc
            inv -= float(daily_demand)
            inv_list.append(inv)
        return inv_list

    rows_current = []
    rows_ordered = []

    for idx, row in df.iterrows():
        sku = row.get("SKU")
        cat = row.get("CAT")
        avail = float(row.get("Available_stock", 0))
        upcoming_qty = float(row.get("Upcoming_stock", 0))
        upcoming_date = row.get("Upcoming_date")
        forecast = float(row.get("Forecast_per_day", 0))
        leadtime = int(row.get("Leadtime_days", 0))
        doc = float(row.get("DOC", 0))

        incoming = {}
        if pd.notna(upcoming_date):
            incoming[pd.Timestamp(upcoming_date).normalize()] = upcoming_qty

        # 1) project current
        proj_current = project_inventory(avail, forecast, incoming, dates)

        # 2) find ROP date
        leadtime_demand = forecast * leadtime
        rop_idx = None
        for i, inv in enumerate(proj_current):
            if inv <= leadtime_demand:
                rop_idx = i
                break
        rop_date = dates[rop_idx] if rop_idx is not None else pd.NaT

        # 3) Order Qty
        order_qty = int(np.ceil(doc * forecast)) if (doc * forecast) > 0 else 0

        if apply_rounding and order_qty > 0:
            if moq and order_qty < moq:
                order_qty = int(moq)
            if pack_size and pack_size > 0:
                order_qty = int(np.ceil(order_qty / pack_size) * pack_size)

        # 4) arrival date
        arrival_date = (pd.to_datetime(rop_date) + pd.Timedelta(days=leadtime)).normalize() if pd.notna(rop_date) and leadtime >= 0 else pd.NaT

        incoming_with_order = dict(incoming)
        if pd.notna(arrival_date) and order_qty > 0:
            incoming_with_order[arrival_date] = incoming_with_order.get(arrival_date, 0) + order_qty

        proj_ordered = project_inventory(avail, forecast, incoming_with_order, dates)

        base_info = {
            "CAT": cat,
            "SKU": sku,
            "Available_stock": avail,
            "Upcoming_stock": upcoming_qty,
            "Upcoming_date": upcoming_date,
            "Forecast_per_day": forecast,
            "Leadtime_days": leadtime,
            "DOC": doc,
            "ROP_date": rop_date,
            "Order_Qty": order_qty,
            "Arrival_date": arrival_date
        }

        for i, d in enumerate(dates):
            day_col = pd.to_datetime(d).strftime("%Y-%m-%d")
            base_info[day_col] = proj_current[i]
        rows_current.append(base_info.copy())

        base_info_ordered = base_info.copy()
        for i, d in enumerate(dates):
            day_col = pd.to_datetime(d).strftime("%Y-%m-%d")
            base_info_ordered[day_col] = proj_ordered[i]
        rows_ordered.append(base_info_ordered)

    output_current = pd.DataFrame(rows_current)
    output_ordered = pd.DataFrame(rows_ordered)

    static_cols = ["CAT","SKU","Available_stock","Upcoming_stock","Upcoming_date","Forecast_per_day","Leadtime_days","DOC","ROP_date","Order_Qty","Arrival_date"]
    date_cols = [c for c in output_current.columns if c not in static_cols]
    date_cols_sorted = sorted(date_cols)
    output_current = output_current[static_cols + date_cols_sorted]
    output_ordered = output_ordered[static_cols + date_cols_sorted]

    st.success(f"Xong — đã tính projection cho {len(output_current)} SKU. Hiển thị 2 bảng Output.current và Output.ordered.")

    with st.expander("🔎 Preview Output.current (first 200 rows)"):
        st.dataframe(output_current.head(200), use_container_width=True)

    with st.expander("🔎 Preview Output.ordered (first 200 rows)"):
        st.dataframe(output_ordered.head(200), use_container_width=True)

    # Download as multi-sheet Excel (xlsxwriter)
    output_bytes = io.BytesIO()
    with pd.ExcelWriter(output_bytes, engine="xlsxwriter") as writer:
        output_current.to_excel(writer, index=False, sheet_name="Output.current")
        output_ordered.to_excel(writer, index=False, sheet_name="Output.ordered")
    st.download_button(
        "⬇️ Tải file Excel (Output.current + Output.ordered)",
        data=output_bytes.getvalue(),
        file_name="Replenishment_Output_from_Input.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("----")
    st.markdown("**Ghi chú**: nếu bạn muốn thay đổi logic OrderQty (ví dụ: target coverage − current), hoặc muốn rounding mặc định theo MOQ/pack, mình sẽ cập nhật theo yêu cầu.")

# End of file
