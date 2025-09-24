import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from io import BytesIO

# ===== Function xử lý =====
def process_file(df_input, days_ahead=30):
    df_input["Upcoming date"] = pd.to_datetime(df_input["Upcoming date"], errors="coerce")

    today = datetime.today().date()
    dates = [today + timedelta(days=i) for i in range(days_ahead + 1)]
    date_cols = [d.strftime("%d-%b-%Y") for d in dates]

    # --- Output.current ---
    output_current = df_input.copy()
    ROP_dates, Order_qtys, stocks_matrix = [], [], []

    for _, row in df_input.iterrows():
        available = row["Available stock"]
        upcoming_stock = row["Upcoming stock"]
        upcoming_date = row["Upcoming date"].date() if pd.notnull(row["Upcoming date"]) else None
        forecast = row["Forecast OB/day"]
        leadtime = int(row["Leadtime (day)"])
        doc = int(row["DOC"])

        stock_list, current_stock = [], available
        oos_date = None

        for d in dates:
            # bán hàng
            current_stock -= forecast
            # cộng thêm incoming
            if upcoming_date and d == upcoming_date:
                current_stock += upcoming_stock
            stock_list.append(current_stock)
            if current_stock <= 0 and oos_date is None:
                oos_date = d

        # ROP date
        if oos_date:
            rop_date = oos_date - timedelta(days=leadtime + 1)
            ROP_dates.append(rop_date.strftime("%d-%b-%Y"))
        else:
            ROP_dates.append(None)

        # Order Qty
        Order_qtys.append(forecast * (leadtime + doc))
        stocks_matrix.append(stock_list)

    output_current["ROP date"] = ROP_dates
    output_current["Order Qty"] = Order_qtys
    df_stock = pd.DataFrame(stocks_matrix, columns=date_cols)
    output_current = pd.concat([output_current, df_stock], axis=1)

    # --- Output.ordered ---
    output_ordered = df_input.copy()
    output_ordered["ROP date"] = ROP_dates
    output_ordered["Order Qty"] = Order_qtys

    stocks_matrix_ordered = []
    for idx, row in df_input.iterrows():
        available = row["Available stock"]
        upcoming_stock = row["Upcoming stock"]
        upcoming_date = row["Upcoming date"].date() if pd.notnull(row["Upcoming date"]) else None
        forecast = row["Forecast OB/day"]
        leadtime = int(row["Leadtime (day)"])
        doc = int(row["DOC"])
        rop_date_str = ROP_dates[idx]
        rop_date = datetime.strptime(rop_date_str, "%d-%b-%Y").date() if rop_date_str else None
        order_qty = Order_qtys[idx]

        stock_list, current_stock = [], available
        for d in dates:
            current_stock -= forecast
            if upcoming_date and d == upcoming_date:
                current_stock += upcoming_stock
            if rop_date and d == rop_date + timedelta(days=leadtime):
                current_stock += order_qty
            stock_list.append(current_stock)
        stocks_matrix_ordered.append(stock_list)

    df_stock_ordered = pd.DataFrame(stocks_matrix_ordered, columns=date_cols)
    output_ordered = pd.concat([output_ordered, df_stock_ordered], axis=1)

    return output_current, output_ordered


def to_excel(df_input, output_current, output_ordered):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_input.to_excel(writer, sheet_name="Input", index=False)
        output_current.to_excel(writer, sheet_name="Output.current", index=False)
        output_ordered.to_excel(writer, sheet_name="Output.ordered", index=False)
    return buffer


# ===== Streamlit App =====
st.title("📦 Hieu Ngan's Planner")

uploaded_file = st.file_uploader("Upload Excel file (sheet Input)", type=["xlsx"])

if uploaded_file:
    df_input = pd.read_excel(uploaded_file, sheet_name="Input")

    st.subheader("📑 Input Data")
    st.dataframe(df_input.head())

    if st.button("🚀 Run Simulation"):
        output_current, output_ordered = process_file(df_input, days_ahead=30)

        st.subheader("✅ Output.current")
        st.dataframe(output_current.head())

        st.subheader("✅ Output.ordered")
        st.dataframe(output_ordered.head())

        result_file = to_excel(df_input, output_current, output_ordered)

        st.download_button(
            label="💾 Download Result Excel",
            data=result_file,
            file_name="Replenishment_Result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
