# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import math

st.markdown("""
<style>
/* === 0. Toàn app: nền hồng nhạt === */
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: #ffe6f0 !important;   /* HỒNG NHẠT */
  color: #111217 !important;
  font-family: "Segoe UI", Roboto, sans-serif;
}

/* === 1. Title === */
h1 {
  color: #111217 !important;
  text-align: center;
  font-weight: 800 !important;
}
h1::after {
  content:"";
  display:block;width:80px;height:3px;margin:10px auto;
  background: linear-gradient(90deg,#ff2f8f,#ff5aa7);
  border-radius:2px;
}

/* === 2. File Uploader Dropzone === */
[data-testid="stFileUploaderDropzone"] {
  background:#111217 !important;     /* ĐEN */
  border:2px dashed #ff5aa7 !important;
  border-radius:14px !important;
  box-shadow:0 4px 12px rgba(255,42,143,.25) !important;
}
[data-testid="stFileUploaderDropzone"] * {
  color:#ffffff !important;          /* chữ TRẮNG */
}
[data-testid="stFileUploader"] button {
  background:#111217 !important;
  color:#ffffff !important;
  border:2px solid #ff5aa7 !important;
  border-radius:10px !important;
}
[data-testid="stFileUploader"] button:hover {
  border-color:#ff2f8f !important;
  background:#1a1a1f !important;
}

/* === 3. File chip (tên file đã chọn) === */
[data-testid="stFileUploadList"] > div {
  background:#ffffff !important;    /* TRẮNG */
  color:#111217 !important;
  border:1px solid #ff5aa7 !important;
  border-left:4px solid #ff2f8f !important;
  border-radius:10px !important;
  padding:6px 10px !important;
}

/* === 4. Alert (success/info/error) === */
.stAlert {
  background:#111217 !important;     /* ĐEN */
  color:#ffffff !important;
  border-left:5px solid #ff5aa7 !important;
  border-radius:10px !important;
  padding:10px !important;
}

/* === 5. Input field === */
.stTextInput>div>div>input,
.stNumberInput input,
.stDateInput>div>div>input,
.stTextArea textarea {
  background:#ffffff !important;     /* TRẮNG */
  color:#111217 !important;
  border:1px solid #ffb6d0 !important;
  border-radius:8px !important;
}
.stTextInput>div>div>input:focus,
.stNumberInput input:focus,
.stDateInput>div>div>input:focus,
.stTextArea textarea:focus {
  border-color:#ff2f8f !important;
}

/* === 6. Dataframe/Table === */
[data-testid="stStyledTable"] {
  background:#111217 !important;    /* ĐEN */
  color:#ffffff !important;
  border:1px solid #ff5aa7 !important;
  border-radius:12px !important;
  overflow:hidden;
}
[data-testid="stStyledTable"] th {
  background:#1a1a1f !important;
  color:#ff5aa7 !important;
  font-weight:800 !important;
}
[data-testid="stStyledTable"] td {
  border-bottom:1px solid #2a2a2a !important;
}
[data-testid="stStyledTable"] tbody tr:nth-child(odd) td {
  background:#161616 !important;
}
[data-testid="stStyledTable"] tbody tr:hover td {
  background:#222 !important;
}

/* === 7. Buttons chung === */
div.stButton > button, .stDownloadButton button {
  background:#111217 !important;  /* ĐEN */
  color:#ffffff !important;       /* chữ TRẮNG */
  border:2px solid #ff5aa7 !important;
  border-radius:12px !important;
  font-weight:700 !important;
  transition:0.2s;
}
div.stButton > button:hover, .stDownloadButton button:hover {
  background:#1c1c1c !important;
  border-color:#ff2f8f !important;
}
</style>
""", unsafe_allow_html=True)



st.set_page_config(page_title="Replenishment HND2025", layout="wide")
st.title("📦 Hieu Ngan's Planner")

# ---------------------------
# Helpers
# ---------------------------
REQUIRED_COLS = [
    "CAT",
    "SKU_code",
    "Available stock",
    "Upcoming stock",
    "Upcoming date",
    "Forecast OB/day",
    "Leadtime (day)",
    "DOC",
]

def validate_input(df):
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return missing

def safe_num(x):
    try:
        return float(x) if not pd.isna(x) else 0.0
    except:
        return 0.0

def process_input(df_input, days_ahead=30, today=None):
    # copy and normalize
    df = df_input.copy()
    df.columns = df.columns.str.strip()
    # parse upcoming date to date
    df["Upcoming date"] = pd.to_datetime(df["Upcoming date"], errors="coerce").dt.date

    # today default
    if today is None:
        today = pd.to_datetime("today").normalize().date()

    # build date range from today to today+days_ahead (inclusive)
    dates = [today + timedelta(days=i) for i in range(days_ahead + 1)]
    # header strings for date columns (ISO format)
    date_cols = [d.strftime("%Y-%m-%d") for d in dates]

    # prepare outputs
    out_current_rows = []
    out_ordered_rows = []

    for idx, row in df.iterrows():
        # read inputs
        cat = row.get("CAT")
        sku = row.get("SKU_code")
        available = safe_num(row.get("Available stock", 0))
        upcoming_stock = safe_num(row.get("Upcoming stock", 0))
        upcoming_date = row.get("Upcoming date")  # is date or NaT
        forecast = safe_num(row.get("Forecast OB/day", 0))
        leadtime = int(safe_num(row.get("Leadtime (day)", 0)))
        doc = safe_num(row.get("DOC", 0))

        # ---------- OUTPUT.CURRENT simulation ----------
        cur = available
        stocks_current = []
        oos_date = None

        # iterate days: for Output.current subtract forecast first, then add incoming if any, then cap to zero
        for d in dates:
            # subtract forecast (end-of-day)
            cur = cur - forecast

            # add upcoming stock if it arrives today
            if (upcoming_date is not None) and (d == upcoming_date):
                cur = cur + upcoming_stock

            # cap at 0 (Excel logic: no negatives shown)
            cur = max(cur, 0.0)

            stocks_current.append(cur)

            # record first OOS day (first day with 0 after capping)
            if (oos_date is None) and (cur == 0.0):
                oos_date = d

        # compute ROP date per your rule: ROP = OOS_date - 1 - leadtime
        if oos_date:
            rop_date = oos_date - timedelta(days=(leadtime + 1))
        else:
            rop_date = None

        # Order Qty: Forecast OB/day * (Leadtime + DOC)
        # ensure DOC numeric (user input may be int)
        order_qty_val = forecast * (leadtime + doc)
        # round up to integer
        order_qty = int(math.ceil(order_qty_val)) if order_qty_val > 0 else 0

        # build base info for this row (columns A->H preserved)
        base = {
            "CAT": cat,
            "SKU_code": sku,
            "Available stock": available,
            "Upcoming stock": upcoming_stock,
            "Upcoming date": upcoming_date,
            "Forecast OB/day": forecast,
            "Leadtime (day)": leadtime,
            "DOC": doc,
            # I, J
            "ROP date": rop_date.strftime("%Y-%m-%d") if rop_date else None,
            "Order Qty": order_qty,
        }

        # prepare output.current row: include daily columns
        cur_row = base.copy()
        for col_name, val in zip(date_cols, stocks_current):
            cur_row[col_name] = val
        out_current_rows.append(cur_row)

        # ---------- OUTPUT.ORDERED simulation ----------
        # For ordered simulation user requested:
        # "stock available của ngày trước đó + upcoming stock (if match) + orderQTY (if day==ROP+leadtime) - Forecast OB/day"
        # that means: add incoming & order at start of the day, then subtract forecast, then cap at 0.
        cur2 = available
        stocks_ordered = []
        arrival_date = (rop_date + timedelta(days=leadtime)) if rop_date else None

        for d in dates:
            # add upcoming if arrives today
            if (upcoming_date is not None) and (d == upcoming_date):
                cur2 = cur2 + upcoming_stock

            # add order if arrival today (ROP_date + leadtime)
            if (arrival_date is not None) and (d == arrival_date):
                cur2 = cur2 + order_qty

            # subtract today's forecast
            cur2 = cur2 - forecast

            # cap
            cur2 = max(cur2, 0.0)

            stocks_ordered.append(cur2)

        ordered_row = base.copy()
        for col_name, val in zip(date_cols, stocks_ordered):
            ordered_row[col_name] = val
        out_ordered_rows.append(ordered_row)

    # Build DataFrames
    output_current = pd.DataFrame(out_current_rows)
    output_ordered = pd.DataFrame(out_ordered_rows)

    # Ensure columns order: A..H (input order) -> ROP date -> Order Qty -> date cols
    input_cols = [c for c in df.columns.tolist()]  # original A..H order
    # Some input columns might be Timestamp objects, normalize names
    # Compose final cols
    final_cols = input_cols + ["ROP date", "Order Qty"] + date_cols
    # Reindex safely (some columns may be missing if input had different order)
    output_current = output_current.reindex(columns=final_cols)
    output_ordered = output_ordered.reindex(columns=final_cols)

    return output_current, output_ordered, date_cols, today

def to_excel_bytes(df_input, out_current, out_ordered):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_input.to_excel(writer, sheet_name="Input", index=False)
        out_current.to_excel(writer, sheet_name="Output.current", index=False)
        out_ordered.to_excel(writer, sheet_name="Output.ordered", index=False)
    buffer.seek(0)
    return buffer

# ---------------------------
# Streamlit UI
# ---------------------------
st.markdown("**Upload** an Excel file that contains only the sheet **'Input'** with columns: "
            "`CAT, SKU_code, Available stock, Upcoming stock, Upcoming date, Forecast OB/day, Leadtime (day), DOC`.")

uploaded = st.file_uploader("Upload Input Excel (.xlsx)", type=["xlsx"])

days_ahead = st.number_input("Simulation days ahead (today included)", min_value=7, max_value=90, value=30, step=1)

if uploaded:
    try:
        df_in = pd.read_excel(uploaded, sheet_name="Input")
    except Exception as e:
        st.error(f"Error reading file or sheet 'Input': {e}")
        st.stop()

    # quick validation
    missing = validate_input(df_in)
    if missing:
        st.error("Missing required columns in your Input sheet: " + ", ".join(missing))
        st.stop()

    st.success("Input read OK. Preview result:")
    st.dataframe(df_in.head(10))

    if st.button("🚀 Run code & Generate Outputs"):
        with st.spinner("Calculating..."):
            out_current, out_ordered, date_cols, used_today = process_input(df_in, days_ahead=int(days_ahead))

        st.subheader("Output.current (preview)")
        st.dataframe(out_current.head(200), use_container_width=True)

        st.subheader("Output.ordered (preview)")
        st.dataframe(out_ordered.head(200), use_container_width=True)

        # prepare excel bytes
        excel_bytes = to_excel_bytes(df_in, out_current, out_ordered)

        st.download_button(
            "⬇️ Download Results",
            data=excel_bytes,
            file_name=f"Replenishment_Result_{used_today}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.success("Hieu Ngan Done — file ready to download.")
else:
    st.info("Please upload an Excel (.xlsx) file that contains a sheet named 'Input'.")
