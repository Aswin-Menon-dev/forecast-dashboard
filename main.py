import os
import requests
import torch
import traceback
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from fpdf import FPDF
from neuralprophet import NeuralProphet

# --- Set page config ---
st.set_page_config(page_title="Samfield Capital Dashboard", layout="wide")

st.markdown("""
<style>
/* ================= MAIN CONTENT ================= */
section.main {
    overflow-x: auto !important;
    overflow-y: auto !important;
    width: 100% !important;
    max-width: 100% !important;
}

section.main > div {
    overflow-x: auto !important;
    overflow-y: auto !important;
    white-space: nowrap !important;
    padding-bottom: 2rem;
}

/* ================= SIDEBAR SCROLL FIX ================= */
section[data-testid="stSidebar"] {
    overflow-y: auto !important;
    overflow-x: hidden !important;
    max-height: 100vh !important;
    scrollbar-width: thin;
    scrollbar-color: #999 transparent;
}

/* Force hide sidebar horizontal scrollbar in Webkit */
section[data-testid="stSidebar"]::-webkit-scrollbar:horizontal {
    display: none !important;
    height: 0 !important;
}

/* ================= SCROLLBAR STYLING ================= */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-thumb {
    background-color: #888;
    border-radius: 4px;
}

/* ================= HEADINGS, TEXT, LOGO ================= */
h1, h2, h3, h4 {
    color: black;
}
.metric-label {
    font-weight: bold;
}
.sidebar-logo {
    transition: none !important;
    transform: none !important;
}

/* ================= RESPONSIVE FIXES ================= */
@media (max-width: 768px) {
    .css-1d391kg {
        padding-left: 10px;
        padding-right: 10px;
    }
    .css-1lcbmhc.e1fqkh3o3 {
        flex-direction: column !important;
    }
    h1 {
        font-size: 1.5rem !important;
    }
    h2 {
        font-size: 1.3rem !important;
    }
}

@media (min-width: 768px) {
    .css-1lcbmhc {
        flex-wrap: nowrap !important;
    }
    .main, section.main {
        flex-grow: 1 !important;
        flex-basis: 0 !important;
        min-width: 0 !important;
    }
}
</style>
""", unsafe_allow_html=True)

# --- Placeholder: Load HR Data (to be replaced with real SenseHR API later) ---
@st.cache_data
def load_hr_data():
    url = "https://api.sensehr.io/employees"  # Placeholder only
    headers = {"Authorization": "Bearer YOUR_API_KEY"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"⚠️ Could not load data from API. Reason: {e}")

        # Fallback to local CSV if available
        if os.path.exists("mock_sensehr_data.csv"):
            st.info("📄 Using local mock HR data.")
            return pd.read_csv("mock_sensehr_data.csv", parse_dates=["Joining Date", "Hire Date", "Application Date"])
        
        # Final fallback: hardcoded dummy data
        st.info("📦 Using built-in dummy HR data.")
        return pd.DataFrame({
            "Employee Name": ["Alice", "Bob", "Charlie", "Diana"],
            "Department": ["HR"] * 4,
            "Joining Date": pd.date_range(start="2023-01-01", periods=4, freq="3M"),
            "Status": ["Active", "Active", "On Leave", "Resigned"],
            "Engagement Score": [78, 82, 75, 80],
            "Training Effectiveness": [76, 80, 79, 77],
            "Performance Rating": [85, 90, 80, 78],
            "Hiring Cost": [4000, 5000, 6000, 5500],
            "Internal Moves": [1, 0, 0, 1],
            "Overtime Hours": [10, 5, 12, 0],
            "Application Status": ["Accepted", "Offered", "Accepted", "Rejected"],
            "Gender": ["Female", "Male", "Female", "Male"],
            "Application Date": pd.date_range("2022-11-01", periods=4, freq="2M"),
            "Hire Date": pd.date_range("2023-01-01", periods=4, freq="2M")
        })

# --- Cached helper for forecasting metrics ---
@st.cache_data
def get_forecast_data(data, forecast_metric):
    df = data.groupby('Week')[forecast_metric].sum().reset_index()
    df = df.rename(columns={'Week': 'ds', forecast_metric: 'y'})
    return df

# --- Sidebar Login and Navigation ---
with st.sidebar:
    st.markdown(
    """
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/200px-React-icon.svg.png" 
         width="150" class="sidebar-logo">
    """,
    unsafe_allow_html=True
)

    st.title("🔐 Login")
    role = st.selectbox("Select your role", ["Admin", "Sales", "HR", "Marketing", "Executive Office"])
    st.markdown("---")
    st.markdown("### 📁 Navigation")
    section = st.radio("Select Department", ["Sales", "HR", "Marketing", "Executive Office"])

# --- Generate Dummy Data ---
np.random.seed(42)
dates = pd.date_range(end=datetime.today(), periods=24, freq='W')
regions = ['North', 'South', 'East', 'West']
data = pd.DataFrame({
    'Week': np.tile(dates, len(regions)),
    'Region': np.repeat(regions, len(dates)),
    'New Leads': np.random.randint(20, 50, size=96),
    'Meetings Booked': np.random.randint(10, 30, size=96),
    'Discovery Calls': np.random.randint(5, 20, size=96),
    'Proposals Sent': np.random.randint(5, 15, size=96),
    'Follow-ups': np.random.randint(10, 25, size=96),
    'Referrals': np.random.randint(2, 10, size=96),
    'CRM Activity': np.random.randint(50, 100, size=96),
    'Content Engagement': np.random.randint(30, 80, size=96),
    'Deals Closed': np.random.randint(5, 20, size=96),
    'Revenue Closed': np.random.randint(1000, 10000, size=96),
    'Sales Cycle Length': np.random.uniform(5, 20, size=96),
    'Conversion Rate': np.random.uniform(0.1, 0.5, size=96),
    'Client Acquisition Cost': np.random.randint(100, 1000, size=96),
    'Forecasted Revenue': np.random.randint(5000, 15000, size=96),
    'New Hires': np.random.randint(0, 10, size=96),
    'Interviews Conducted': np.random.randint(5, 20, size=96),
    'Employee Turnover': np.random.randint(0, 5, size=96),
    'Engagement Score': np.random.uniform(60, 90, size=96),
    'Training Hours': np.random.randint(10, 50, size=96),
    'Job Applications': np.random.randint(50, 150, size=96),
    'Website Traffic': np.random.randint(1000, 10000, size=96),
    'Social Media Engagement': np.random.randint(200, 1000, size=96),
    'Campaigns Run': np.random.randint(1, 5, size=96),
    'Lead Conversion': np.random.uniform(0.1, 0.6, size=96),
    'MQLs Generated': np.random.randint(10, 100, size=96),
    'Strategic Projects': np.random.randint(1, 3, size=96),
    'Weekly Objectives Met': np.random.randint(2, 5, size=96),
    'Cross-team Meetings': np.random.randint(3, 10, size=96),
    'Compliance Score': np.random.uniform(85, 100, size=96)
})

# --- Filters ---
st.sidebar.markdown("### 🌍 Filter by Region")
regions = sorted(data['Region'].unique())
selected_region = st.sidebar.selectbox("Choose Region", ["All"] + regions)

st.sidebar.markdown("### 📆 Filter by Date")
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(weeks=8))
end_date = st.sidebar.date_input("End Date", datetime.today())
data = data[(data['Week'] >= pd.to_datetime(start_date)) & (data['Week'] <= pd.to_datetime(end_date))]
if selected_region != "All":
    data = data[data['Region'] == selected_region]

# --- Saved Views ---
st.sidebar.markdown("### 💾 Saved Views")
saved_view = st.sidebar.selectbox("Load Saved View", ["None", "Sales - South", "HR - West"])
if saved_view == "Sales - South":
    section = "Sales"
    selected_region = "South"
elif saved_view == "HR - West":
    section = "HR"
    selected_region = "West"

# --- KPI Targets ---
if 'targets' not in st.session_state:
    st.session_state.targets = {
        'New Leads': 40,
        'Deals Closed': 15,
        'Revenue Closed': 7000,
        'New Hires': 5,
        'Website Traffic': 5000,
        'Strategic Projects': 2
    }

if role == "Admin":
    st.sidebar.markdown("### 🎯 Edit KPI Targets")
    for metric in st.session_state.targets:
        new_val = st.sidebar.number_input(f"Target - {metric}", value=int(st.session_state.targets[metric]), step=1)
        st.session_state.targets[metric] = new_val

# --- Theme and Heading ---
themes = {"Sales": "#1f77b4", "HR": "#2ca02c", "Marketing": "#d62728", "Executive Office": "#9467bd"}
st.markdown(f"<h1 style='color:black;'>{section} Dashboard</h1>", unsafe_allow_html=True)

# --- Metric Display with Target ---
def display_metric_with_target(col, metric, label):
    current = data[metric].iloc[-1]
    previous = data[metric].iloc[-2] if len(data) > 1 else current
    change = current - previous
    alert = "📍 Spike" if previous and (change / previous > 0.25) else "⚠️ Drop" if previous and (change / previous < -0.25) else ""
    if metric in st.session_state.targets:
        delta = float(current - st.session_state.targets[metric])
        col.metric(f"{label} {alert}", f"{int(current)} / {st.session_state.targets[metric]}", delta=delta)
    else:
        col.metric(f"{label} {alert}", int(current))

# --- Department Dashboards ---
if section == "Sales" and role in ["Admin", "Sales"]:
    with st.spinner("Loading Sales Dashboard..."):
        scrollable = st.container()
        with scrollable:
            st.subheader(f"🟦 {section} Metrics")
            lead_sales = [
            ("New Leads", "Number of New Leads/Prospects Added"),
            ("Meetings Booked", "Client Meetings Booked"),
            ("Discovery Calls", "Discovery Calls Conducted"),
            ("Proposals Sent", "Proposals/Quotes Sent"),
            ("Follow-ups", "Follow-ups Made"),
            ("Referrals", "Referral or Partner Introductions"),
            ("CRM Activity", "CRM Activity Logged"),
            ("Content Engagement", "Marketing/Content Engagement")
            ]
        cols = st.columns(4)
        for i, (metric, label) in enumerate(lead_sales):
            display_metric_with_target(cols[i % 4], metric, label)

        st.subheader("🟩 Lag Indicators")
        lag_sales = [
            ("Deals Closed", "Deals Closed"),
            ("Revenue Closed", "Revenue Closed"),
            ("Sales Cycle Length", "Sales Cycle Length"),
            ("Conversion Rate", "Conversion Rates"),
            ("Client Acquisition Cost", "Client Acquisition Cost"),
            ("Forecasted Revenue", "Forecasted Revenue")
        ]
        cols = st.columns(3)
        for i, (metric, label) in enumerate(lag_sales):
            display_metric_with_target(cols[i % 3], metric, label)

elif section == "HR" and role in ["Admin", "HR"]:
    with st.spinner("Loading HR Dashboard..."):
        st.subheader("🟦 HR Metrics")
        hr_data = load_hr_data()

        # --- Ensure date columns are proper datetime ---
        for col in ['Joining Date', 'Hire Date', 'Application Date']:
            if col in hr_data.columns:
                hr_data[col] = pd.to_datetime(hr_data[col], errors='coerce')

        total_employees = len(hr_data[hr_data['Status'].isin(["Active", "On Leave"])])
        resigned_employees = len(hr_data[hr_data['Status'] == 'Resigned'])

        # --- Core Metrics ---
        cost_per_hire = hr_data["Hiring Cost"].mean() if "Hiring Cost" in hr_data.columns else 5000

        if {'Application Date', 'Hire Date'}.issubset(hr_data.columns):
            hr_data["Time to Hire"] = (hr_data["Hire Date"] - hr_data["Application Date"]).dt.days
            time_to_hire = hr_data["Time to Hire"].mean()
        else:
            time_to_hire = 15

        turnover_rate = resigned_employees / len(hr_data) * 100 if len(hr_data) else 0
        retention_rate = 100 - turnover_rate
        revenue_per_employee = data["Revenue Closed"].iloc[-1] / total_employees if total_employees else 0

        if "Application Status" in hr_data.columns:
            offers = len(hr_data[hr_data["Application Status"] == "Offered"])
            accepted = len(hr_data[hr_data["Application Status"] == "Accepted"])
            acceptance_rate = accepted / offers * 100 if offers else 0
        else:
            acceptance_rate = 70.0

        # --- Diversity & Promotions ---
        internal_promotion_rate = hr_data["Internal Moves"].sum() / total_employees * 100 if total_employees else 0
        headcount = total_employees
        overtime_expense = hr_data["Overtime Hours"].sum() * 250 if "Overtime Hours" in hr_data.columns else 25000

        # --- Engagement & Training ---
        training_effectiveness = hr_data["Training Effectiveness"].mean() if "Training Effectiveness" in hr_data.columns else 78
        performance_score = hr_data["Performance Rating"].mean() if "Performance Rating" in hr_data.columns else 82
        satisfaction_score = hr_data["Engagement Score"].mean() if "Engagement Score" in hr_data.columns else 75

        # --- Display Core HR Metrics ---
        st.subheader("📊 Core HR Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Cost per Hire", f"${cost_per_hire:,.0f}")
        col2.metric("Time to Hire", f"{time_to_hire:.1f} days")
        col3.metric("Turnover Rate", f"{turnover_rate:.1f}%")

        col4, col5, col6 = st.columns(3)
        col4.metric("Retention Rate", f"{retention_rate:.1f}%")
        col5.metric("Revenue per Employee", f"${revenue_per_employee:,.0f}")
        col6.metric("Acceptance Rate", f"{acceptance_rate:.1f}%")

        # --- Diversity & Structure ---
        st.subheader("🌍 Diversity & Structure")
        col7, col8, col9 = st.columns(3)
        col7.metric("Internal Promotion Rate", f"{internal_promotion_rate:.1f}%")
        col8.metric("Headcount", headcount)
        col9.metric("Overtime Expense", f"${overtime_expense:,.0f}")

        # --- Engagement & Training ---
        st.subheader("📘 Engagement & Training")
        col10, col11, col12 = st.columns(3)
        col10.metric("Training Effectiveness", f"{training_effectiveness:.1f}%")
        col11.metric("Employee Performance", f"{performance_score:.1f}%")
        col12.metric("Satisfaction Score", f"{satisfaction_score:.1f}")

        # --- Gender Diversity Chart ---
        if "Gender" in hr_data.columns:
            st.subheader("🧬 Gender Diversity")
            gender_count = hr_data["Gender"].value_counts().reset_index()
            gender_count.columns = ["Gender", "Count"]
            fig = px.pie(gender_count, names="Gender", values="Count", title="Gender Distribution")
            st.plotly_chart(fig, use_container_width=True)

        # --- Employee Status Chart ---
        st.subheader("🥧 Employee Status Distribution")
        status_count = hr_data["Status"].value_counts().reset_index()
        status_count.columns = ["Status", "Count"]
        fig = px.pie(status_count, names="Status", values="Count", title="Employee Status Breakdown")
        st.plotly_chart(fig, use_container_width=True)

        # --- Monthly Revenue per Employee ---
        st.subheader("📈 Monthly Revenue per Employee")
        monthly_revenue = data.groupby(data['Week'].dt.to_period("M")).agg({
            "Revenue Closed": "sum"
        }).reset_index()
        monthly_revenue["Month"] = monthly_revenue["Week"].astype(str)
        monthly_revenue["Employees"] = total_employees
        monthly_revenue["Revenue per Employee"] = monthly_revenue["Revenue Closed"] / total_employees

        fig = px.line(monthly_revenue, x="Month", y="Revenue per Employee",
                      title="Monthly Revenue per Employee", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        # --- Monthly Absenteeism Rate ---
        st.subheader("📈 Monthly Absenteeism Rate")
        if "Joining Date" in hr_data.columns:
            hr_data["Month"] = hr_data["Joining Date"].dt.to_period("M")
            absent_by_month = hr_data.groupby("Month").apply(
                lambda df: (df["Status"] == "On Leave").sum() / len(df) * 100
            ).reset_index(name="Absenteeism Rate")
            absent_by_month["Month"] = absent_by_month["Month"].astype(str)

            fig = px.line(absent_by_month, x="Month", y="Absenteeism Rate",
                          title="Monthly Absenteeism Rate", markers=True)
            st.plotly_chart(fig, use_container_width=True)

        st.download_button("⬇️ Download HR Data as CSV", hr_data.to_csv(index=False), file_name="hr_employee_records.csv")

        st.info("🔗 This is demo HR data. Live integration will replace dummy data once API access is connected.")

elif section == "Marketing" and role in ["Admin", "Marketing"]:
    with st.spinner("Loading Marketing Dashboard..."):
        st.subheader("🟦 Marketing Metrics")
        marketing_metrics = [
            ("Website Traffic", "Website Traffic"),
            ("Social Media Engagement", "Social Media Engagement"),
            ("Campaigns Run", "Campaigns Run"),
            ("Lead Conversion", "Lead Conversion Rate"),
            ("MQLs Generated", "MQLs Generated")
        ]

        chart_type_marketing = st.selectbox("📊 Choose chart type for Marketing Metrics:",
                                            ["Bar", "Line", "Histogram", "Pie"],
                                            key="marketing_chart")

        for metric, label in marketing_metrics:
            st.markdown(f"**{label}**")
            grouped = data.groupby("Week")[metric].sum().reset_index()

            if chart_type_marketing == "Bar":
                fig = px.bar(grouped, x="Week", y=metric, title=label)
            elif chart_type_marketing == "Line":
                fig = px.line(grouped, x="Week", y=metric, markers=True, title=label)
            elif chart_type_marketing == "Histogram":
                fig = px.histogram(data, x=metric, title=f"Histogram of {label}")
            elif chart_type_marketing == "Pie":
                latest_week = data['Week'].max()
                pie_data = data[data['Week'] == latest_week].groupby("Region")[metric].sum().reset_index()
                fig = px.pie(pie_data, names="Region", values=metric, title=f"{label} Distribution by Region")

            st.plotly_chart(fig, use_container_width=True)

        # Download CSV
        st.download_button("⬇️ Download Marketing Data as CSV", data.to_csv(index=False), file_name="marketing_data.csv")
        st.info("🔗 This is demo HR data. Real-time data will be fetched from SenseHR API once access is available.")

elif section == "Executive Office" and role in ["Admin", "Executive Office"]:
    with st.spinner("Loading Executive Overview..."):
        st.subheader("🟦 Executive KPIs")
        exec_metrics = [
            ("Strategic Projects", "Strategic Projects Completed"),
            ("Weekly Objectives Met", "Weekly Objectives Met"),
            ("Cross-team Meetings", "Cross-team Meetings"),
            ("Compliance Score", "Compliance Score")
        ]
        cols = st.columns(2)
        for i, (metric, label) in enumerate(exec_metrics):
            display_metric_with_target(cols[i % 2], metric, label)

else:
    st.warning("🚫 You do not have access to this section.")

# --- Executive Summary ---
st.subheader("📝 Executive Summary")
summaries = []
for metric in ["Revenue Closed", "New Leads", "Deals Closed"]:
    if len(data) > 1:
        prev_value = data[metric].iloc[-2]
        diff = data[metric].iloc[-1] - prev_value
        pct = round((diff / prev_value) * 100, 1) if prev_value else 0
        direction = "increased" if diff > 0 else "decreased"
        summaries.append(f"{metric} {direction} by {abs(pct)}% compared to last week.")
for line in summaries:
    st.write(f"- {line}")

# --- Regional Metric Chart with Dropdown ---
st.subheader("📊 Regional Metric Chart")
chart_type_region = st.selectbox(
    "Choose chart type for Regional Metric Chart:",
    ["Bar Chart", "Line Chart", "Area Chart", "Histogram", "Pie Chart"],
    key="region_chart_type"
)
selected_metric_region = st.selectbox(
    "Select a metric to display by region:",
    ["Revenue Closed", "New Leads", "Deals Closed"],
    key="region_metric"
)

agg_data_region = data.groupby(["Region", "Week"])[selected_metric_region].sum().reset_index()

if chart_type_region == "Bar Chart":
    agg_sum_region = agg_data_region.groupby("Region")[selected_metric_region].sum().reset_index()
    fig = px.bar(
        agg_sum_region,
        x='Region',
        y=selected_metric_region,
        color='Region',
        color_discrete_sequence=px.colors.qualitative.Set2,
        title=f"Total {selected_metric_region} by Region"
    )
    st.plotly_chart(fig, use_container_width=True)

elif chart_type_region == "Line Chart":
    fig = px.line(
        agg_data_region,
        x='Week',
        y=selected_metric_region,
        color='Region',
        markers=True,
        title=f"{selected_metric_region} over Time by Region"
    )
    st.plotly_chart(fig, use_container_width=True)

elif chart_type_region == "Area Chart":
    fig = px.area(
        agg_data_region,
        x='Week',
        y=selected_metric_region,
        color='Region',
        title=f"{selected_metric_region} Area Chart by Region"
    )
    st.plotly_chart(fig, use_container_width=True)

elif chart_type_region == "Histogram":
    fig = px.histogram(
        data,
        x=selected_metric_region,
        color='Region',
        barmode='overlay',
        title=f"Histogram of {selected_metric_region} by Region"
    )
    st.plotly_chart(fig, use_container_width=True)

elif chart_type_region == "Pie Chart":
    agg_sum_region = agg_data_region.groupby("Region")[selected_metric_region].sum().reset_index()
    fig = px.pie(
        agg_sum_region,
        names='Region',
        values=selected_metric_region,
        title=f"{selected_metric_region} Distribution by Region"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Multi-Metric Comparison with Dropdown ---
st.subheader("📊 Multi-Metric Comparison")
chart_type_multi = st.selectbox(
    "Choose chart type for Multi-Metric Comparison:",
    ["Bar Chart", "Line Chart", "Area Chart", "Histogram", "Pie Chart"],
    key="multi_chart_type"
)

selected_metrics_multi = st.multiselect(
    "Select metrics to compare:",
    options=list(data.columns[2:]),
    default=["New Leads", "Deals Closed"],
    key="multi_metrics"
)

if selected_metrics_multi:
    if chart_type_multi == "Bar Chart":
        agg_data_multi = data.groupby("Region")[selected_metrics_multi].sum().reset_index()
        fig = go.Figure()
        for metric in selected_metrics_multi:
            fig.add_trace(go.Bar(
                x=agg_data_multi['Region'],
                y=agg_data_multi[metric],
                name=metric
            ))
        fig.update_layout(
            barmode='group',
            title="Sum of Selected Metrics by Region",
            xaxis_title="Region",
            yaxis_title="Value"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type_multi == "Line Chart":
        fig = px.line(
            data,
            x='Week',
            y=selected_metrics_multi,
            color='Region',
            markers=True,
            title="Selected Metrics Over Time by Region"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type_multi == "Area Chart":
        fig = px.area(
            data,
            x='Week',
            y=selected_metrics_multi[0],  # Only first metric shown
            color='Region',
            title=f"{selected_metrics_multi[0]} - Area Chart by Region"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type_multi == "Histogram":
        fig = px.histogram(
            data,
            x=selected_metrics_multi[0],
            color='Region',
            barmode='overlay',
            title=f"Histogram of {selected_metrics_multi[0]} by Region"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type_multi == "Pie Chart":
        pie_data = data.groupby("Region")[selected_metrics_multi[0]].sum().reset_index()
        fig = px.pie(
            pie_data,
            names='Region',
            values=selected_metrics_multi[0],
            title=f"{selected_metrics_multi[0]} Distribution by Region"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Forecasting & Predictive Analytics ---
st.subheader("🔮 Forecasting & Predictive Analytics")
st.write("📈 Predict future performance trends with NeuralProphet.")

forecast_metric = st.selectbox(
    "Select metric for forecasting:",
    ["Revenue Closed", "New Leads", "Deals Closed", "DEBUG"],
    key="forecast_metric"
)

chart_type_forecast = st.radio(
    "Select chart type:",
    ["Line Chart", "Bar Chart"],
    key="forecast_chart"
)

# Special override for DEBUG mode
if forecast_metric == "DEBUG":
    forecast_data = pd.DataFrame({
        'ds': pd.date_range(start='2023-01-01', periods=24, freq='W'),
        'y': np.random.randint(1000, 5000, 24)
    })
else:
    forecast_data = get_forecast_data(data, forecast_metric)

# Clean the data to avoid NaN or non-float issues
forecast_data = forecast_data.dropna(subset=['ds', 'y']).copy()
forecast_data['y'] = pd.to_numeric(forecast_data['y'], errors='coerce').fillna(0).astype(float)

# Check if forecast_data is valid
if forecast_data.empty or 'y' not in forecast_data.columns or forecast_data['y'].sum() == 0:
    st.warning(f"⚠️ No valid data available for forecasting '{forecast_metric}'. Try selecting a different metric or adjust filters.")
    st.stop()  # prevent further execution

# --- Forecast Data Block (DEBUG + Real) ---
if forecast_metric == "DEBUG":
    forecast_data = pd.DataFrame({
        'ds': pd.date_range(start='2023-01-01', periods=24, freq='W'),
        'y': np.random.randint(1000, 5000, 24)
    })
else:
    forecast_data = get_forecast_data(data, forecast_metric)

# Clean the data to avoid NaN or non-float issues
forecast_data = forecast_data.dropna(subset=['ds', 'y']).copy()
forecast_data['y'] = pd.to_numeric(forecast_data['y'], errors='coerce').fillna(0).astype(float)

# Show only one preview
st.write("🧪 Forecast Data Preview", forecast_data.head())
st.write("📏 Shape:", forecast_data.shape)

# Validation
if forecast_data.empty or 'y' not in forecast_data.columns or forecast_data['y'].sum() == 0:
    st.warning(f"⚠️ No valid data available for forecasting '{forecast_metric}'. Try DEBUG or adjust filters.")
    st.stop()

@st.cache_resource
def train_forecast_model(forecast_data):
    model = NeuralProphet(n_changepoints=10, yearly_seasonality=False, weekly_seasonality=True)
    model.fit(forecast_data, freq='W')
    return model

try:
    model = train_forecast_model(forecast_data)
    future = model.make_future_dataframe(forecast_data, periods=8)
    forecast = model.predict(future)

    st.markdown("## 🔮 Forecasting Output")

    if chart_type_forecast == "Line Chart":
        fig_forecast = px.line(forecast, x='ds', y='yhat1', title=f"{forecast_metric} Forecast")
        fig_forecast.add_scatter(x=forecast_data['ds'], y=forecast_data['y'], mode='markers', name='Actual')
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        fig_bar = px.bar(forecast, x='ds', y='yhat1', title=f"{forecast_metric} Forecast (Bar)")
        st.plotly_chart(fig_bar, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ Forecasting failed: {e}")
    import traceback
    st.code(traceback.format_exc())

# Forecasting Logic
if forecast_data.empty or 'y' not in forecast_data.columns or forecast_data['y'].sum() == 0:
    st.warning(f"⚠️ No data available to forecast '{forecast_metric}'. Try DEBUG mode or adjust filters.")
    st.stop()

try:
    @st.cache_resource
    def train_forecast_model(forecast_data):
        model = NeuralProphet(n_changepoints=10, yearly_seasonality=False, weekly_seasonality=True)
        model.fit(forecast_data, freq='W')
        return model

    model = train_forecast_model(forecast_data)
    future = model.make_future_dataframe(forecast_data, periods=8)
    forecast = model.predict(future)

except Exception as e:
    st.error(f"⚠️ Forecasting failed: {e}")
    import traceback
    st.code(traceback.format_exc())


# --- Notes Section ---
if role == "Admin":
    st.subheader("🗨️ Notes & Context")
    notes = st.text_area("Add notes for this week's data:")
    if notes:
        st.success("Note saved (in memory)")

# --- Download Data ---
st.subheader("⬇️ Download Data")
st.download_button("Download CSV", data.to_csv(index=False), file_name="dashboard_data.csv")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align:center; font-size: 14px;'>This is a prototype (Developed by Aswin Menon)</p>", unsafe_allow_html=True)
