import io
import pandas as pd
import streamlit as st
from datetime import timedelta

st.set_page_config(page_title="Inbound Planner", layout="wide")

st.title("📦 Hieu Ngan's Planner")

# Upload file Excel
excel_file = st.file_uploader("Upload Excel file (Replenishment Auto.xlsx)", type=["xlsx"])

if excel_file:
    # Đọc sheet Input
    df_input = pd.read_excel(excel_file, sheet_name="Input")

    st.subheader("📥 Input data")
    st.dataframe(df_input, use_container_width=True)

    # --- Output.current ---
    df_current = df_input.copy()

    # Demo logic: ROP date = hôm nay + leadtime, Order Qty = forecast * DOC
    df_current["ROP date"] = pd.to_datetime("today") + pd.to_timedelta(df_current["Leadtime (day)"], unit="D")
    df_current["Order Qty"] = (df_current["Forecast OB/day"] * df_current["DOC"]).astype(int)

    # --- Output.ordered ---
    df_ordered = df_current.copy()

    horizon = 14  # giả định 14 ngày forecast
    for d in range(1, horizon + 1):
        date_col = (pd.to_datetime("today") + timedelta(days=d)).strftime("%Y-%m-%d")
        df_ordered[date_col] = (
            df_ordered["Available stock"]
            + df_ordered["Upcoming stock"]
            + df_ordered["Order Qty"]
            - df_ordered["Forecast OB/day"] * d
        )

    # Hiển thị kết quả
    st.subheader("📊 Output.current")
    st.dataframe(df_current, use_container_width=True)

    st.subheader("📊 Output.ordered")
    st.dataframe(df_ordered, use_container_width=True)

    # Xuất Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_current.to_excel(writer, index=False, sheet_name="Output.current")
        df_ordered.to_excel(writer, index=False, sheet_name="Output.ordered")

    st.download_button(
        "⬇️ Download Excel Output",
        data=output.getvalue(),
        file_name="Replenishment_Output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("👉 Vui lòng upload file Excel theo format của sheet 'Input'.")
