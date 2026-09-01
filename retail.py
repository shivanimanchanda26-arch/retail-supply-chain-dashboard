import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page Configuration (Must be the very first Streamlit command)
st.set_page_config(page_title="Retail Supply Chain 360", layout="wide")

st.title("📦 Retail Supply Chain 360-Degree Analytics Dashboard")
st.markdown("Comprehensive view of store inventory, logistics performance, supplier metrics, and operational bottlenecks.")

# --- SYNTHETIC DATA GENERATION ---
@st.cache_data
def generate_supply_chain_data():
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", end="2026-08-31", freq="D")
    
    regions = ["North America", "Europe", "Asia-Pacific", "Latin America"]
    categories = ["Electronics", "Apparel", "Home Goods", "Groceries", "Beauty"]
    vendors = ["Vendor Alpha", "Vendor Beta", "Vendor Gamma", "Vendor Delta", "Vendor Omega"]
    
    data = []
    for store_id in range(1, 11):
        region = np.random.choice(regions)
        for date in np.random.choice(dates, size=20):
            category = np.random.choice(categories)
            vendor = np.random.choice(vendors)
            demand = np.random.randint(100, 1000)
            actual_sales = int(demand * np.random.uniform(0.85, 1.05))
            inventory_level = np.random.randint(50, 1500)
            lead_time = np.random.randint(3, 21)
            otif = np.random.uniform(75.0, 99.0)
            defect_rate = np.random.uniform(0.5, 4.5)
            
            data.append([
                date, f"Store_{store_id:03d}", region, category, vendor,
                demand, actual_sales, inventory_level, lead_time, otif, defect_rate
            ])
            
    df = pd.DataFrame(data, columns=[
        "Date", "Store_ID", "Region", "Category", "Vendor",
        "Demand", "Actual_Sales", "Inventory_Level", "Lead_Time_Days", "OTIF_Rate", "Defect_Rate"
    ])
    return df

try:
    df = generate_supply_chain_data()
except Exception as e:
    st.error(f"Error generating data: {e}")

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔍 Dashboard Filters")
selected_region = st.sidebar.selectbox("Select Region", ["All"] + list(df["Region"].unique()))
selected_category = st.sidebar.selectbox("Select Category", ["All"] + list(df["Category"].unique()))

filtered_df = df.copy()
if selected_region != "All":
    filtered_df = filtered_df[filtered_df["Region"] == selected_region]
if selected_category != "All":
    filtered_df = filtered_df[filtered_df["Category"] == selected_category]

# --- KPI SUMMARY CARDS ---
st.markdown("### 📊 Executive Performance KPIs")
col1, col2, col3, col4, col5 = st.columns(5)

total_revenue = filtered_df["Actual_Sales"].sum() * 45
avg_otif = filtered_df["OTIF_Rate"].mean() if not filtered_df.empty else 0
avg_lead_time = filtered_df["Lead_Time_Days"].mean() if not filtered_df.empty else 0
avg_defect = filtered_df["Defect_Rate"].mean() if not filtered_df.empty else 0
stockout_risk_count = len(filtered_df[filtered_df["Inventory_Level"] < 100])

col1.metric("Est. Revenue", f"${total_revenue:,.0f}")
col2.metric("Avg OTIF Rate", f"{avg_otif:.1f}%")
col3.metric("Avg Lead Time", f"{avg_lead_time:.1f} Days")
col4.metric("Avg Defect Rate", f"{avg_defect:.1f}%")
col5.metric("Low Stock Alerts", f"{stockout_risk_count}")

st.markdown("---")

# --- TABS FOR 360 VIEW ---
tab1, tab2, tab3 = st.tabs(["📈 Operational Overview", "🚚 Supplier & Logistics", "📋 Store-Level Details & AI Insights"])

with tab1:
    st.subheader("Demand Forecasting vs. Actual Sales")
    if not filtered_df.empty:
        time_series = filtered_df.groupby("Date")[["Demand", "Actual_Sales"]].sum().reset_index()
        fig_ts = px.line(time_series, x="Date", y=["Demand", "Actual_Sales"], title="Demand vs Actual Sales Trends Over Time")
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.warning("No data available for selected filters.")
    
    st.subheader("Inventory Levels by Product Category")
    if not filtered_df.empty:
        inv_cat = filtered_df.groupby("Category")["Inventory_Level"].sum().reset_index()
        fig_inv = px.bar(inv_cat, x="Category", y="Inventory_Level", color="Category", title="Total Stock Volume per Category")
        st.plotly_chart(fig_inv, use_container_width=True)

with tab2:
    st.subheader("Vendor Performance: Lead Time vs. Defect Rate")
    if not filtered_df.empty:
        vendor_perf = filtered_df.groupby("Vendor")[["Lead_Time_Days", "Defect_Rate", "OTIF_Rate"]].mean().reset_index()
        fig_vendor = px.scatter(vendor_perf, x="Lead_Time_Days", y="Defect_Rate", size="OTIF_Rate", color="Vendor",
                                hover_name="Vendor", title="Vendor Evaluation (Bubble size represents OTIF Rate)")
        st.plotly_chart(fig_vendor, use_container_width=True)
    else:
        st.warning("No vendor data available.")

with tab3:
    st.subheader("Store-Level Performance Data Grid")
    st.dataframe(filtered_df.head(50), use_container_width=True)
    
    st.subheader("🤖 Automated Supply Chain AI Insights")
    st.info(
        "- **Bottleneck Detected:** Vendors with lead times exceeding 14 days are driving up local safety stock requirements.\n"
        "- **Stockout Warning:** Electronics and Home Goods categories show higher instances of inventory dips below the safety threshold.\n"
        "- **Recommendation:** Consider shifting procurement volume toward vendors maintaining an OTIF rate above 90% to stabilize fulfillment pipelines."
    )
