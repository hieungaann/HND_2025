import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="📦 Replenishment Planner", layout="wide")

st.title("📦 Hieu Ngan's Replenishment Planner")

uploaded = st.file_uploader("Upload Excel (Replenishment Auto.xlsx)", type=["xlsx"])

if uploaded:
    # Đọc sheet Input
    df_input = pd.read_excel(uploaded, sheet_name="Input")

    # Xác định hôm nay
    today = datetime.today().date()

    # Clone lại input cho output
    df_out = df_input.copy()

    # Đảm bảo format ngày cho Upcoming date
    if "Upcoming date" in df_out.columns:
        df_out["Upcoming date"] = pd.to_datetime(df_out["Upcoming date"], errors="coerce").dt.date

    # Tạo các cột ngày (today → today+30)
    date_cols = [(today + timedelta(days=i)) for i in range(31)]

    # Khởi tạo stock projection
    proj = []
    for _, row in df_out.iterrows():
        daily_fc = row["Forecast OB/day"]
        stock = row["Available stock"]

        sku_proj = {}
        for d in date_cols:
            # Trừ forecast trước (cuối ngày)
            stock -= daily_fc

            # Nếu có inbound đúng ngày này → cộng thêm
            if pd.notna(row.get("Upcoming stock", None)) and pd.notna(row.get("Upcoming date", None)):
                if d == row["Upcoming date"]:
                    stock += row["Upcoming stock"]

            # Lưu stock cuối ngày
            sku_proj[d] = max(stock, 0)

        proj.append(sku_proj)

    df_proj = pd.DataFrame(proj)
    df_proj.columns = [d.strftime("%d-%b") for d in df_proj.columns]  # format cột ngày đẹp

    # Tính ROP date
    rop_dates = []
    for idx, row in df_proj.iterrows():
        first_zero = None
        for d in df_proj.columns:
            if row[d] <= 0:
                # parse date từ tên cột
                first_zero = datetime.strptime(d + f"-{today.year}", "%d-%b-%Y").date()
                break
        if first_zero:
            leadtime = int(df_out.loc[idx, "Leadtime (day)"])
            rop_dates.append(first_zero - timedelta(days=leadtime))
        else:
            rop_dates.append(None)

    df_out["ROP date"] = rop_dates

    # Tính Order Qty
    df_out["Order Qty"] = df_out["Forecast OB/day"] * df_out["DOC"]

    # Ghép kết quả cuối
    result = pd.concat([df_out, df_proj], axis=1)

    st.success(f"✅ Đã tính xong Output.current cho {len(result)} SKU")
    st.dataframe(result, use_container_width=True)

    # Xuất Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        result.to_excel(writer, sheet_name="Output.current", index=False)
    st.download_button(
        "⬇️ Tải Output.current",
        data=output.getvalue(),
        file_name="Output.current.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

else:
    st.info("👉 Hãy upload file Excel `Replenishment Auto.xlsx` để bắt đầu.")
