import streamlit as st
import pandas as pd
import plotly.express as px
from pyhive import hive

# Set the page configuration
st.set_page_config(
    page_title="E-Commerce Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configure the Hive connection
hive_host = "hive-server"
hive_port = 10000
hive_database = "ecommerce_db"

# Establish Hive connection
@st.cache_resource
def get_hive_connection():
    return hive.Connection(host=hive_host, port=hive_port, database=hive_database)

# Query Hive to get data
@st.cache_data
def get_hive_data(query):
    conn = get_hive_connection()
    return pd.read_sql(query, conn)

# Load the data from Hive table
@st.cache_data
def load_table_data():
    query = "SELECT * FROM ecommerce_transformed"
    data = get_hive_data(query)
    data.columns = [col.split('.')[-1] for col in data.columns]  # Simplify column names
    data['invoicedate'] = pd.to_datetime(data['invoicedate'], errors='coerce')  # Convert dates
    return data


# Load data
data = load_table_data()

# Custom CSS for modern design
st.markdown("""
    <style>
        /* Set custom background color */
        .main {
            background-color: #f5f7fa;
        }

        /* Unique sidebar styling */
        section[data-testid="stSidebar"] {
            color: black;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] h5,
        section[data-testid="stSidebar"] h6 {
            color: #000000;
        }

        /* Adjust text and card styles */
        .stMetricLabel {
            color: #2c3e50;
        }
        .stMarkdown {
            margin-bottom: 15px;
        }

    </style>
""", unsafe_allow_html=True)

# Title
st.title("📊 E-Commerce Sales Dashboard")

# Top KPIs
st.markdown("### Key Metrics")
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric(label="🛒 Total Transactions", value=data.shape[0], delta=None)

with kpi2:
    total_quantity = data["quantity"].sum()
    st.metric(label="📦 Total Quantity Sold", value=f"{total_quantity:,}")

with kpi3:
    total_revenue = (data["quantity"] * data["unitprice"]).sum()
    st.metric(label="💰 Total Revenue (€)", value=f"€{total_revenue:,.2f}")

# Sidebar for Filters
st.sidebar.header("Filter Data")
st.sidebar.markdown("Apply filters to customize the data view.")

country_filter = st.sidebar.multiselect(
    "🌍 Select Countries:",
    options=data['country'].dropna().unique(),
    default=None
)

min_date, max_date = data['invoicedate'].min(), data['invoicedate'].max()
date_filter = st.sidebar.date_input(
    "📅 Date Range:",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date,
)

# Filter Data
filtered_data = data[
    (data['country'].isin(country_filter)) &
    (data['invoicedate'] >= pd.Timestamp(date_filter[0])) &
    (data['invoicedate'] <= pd.Timestamp(date_filter[1]))
]

# Filtered Data Preview
st.markdown("### Filtered Data Preview")
st.dataframe(filtered_data.head(), use_container_width=True)

# Download Filtered Data
st.download_button(
    label="⬇️ Download Filtered Data",
    data=filtered_data.to_csv(index=False).encode('utf-8'),
    file_name='filtered_data.csv',
    mime='text/csv',
)

# Visualizations
st.markdown("### Visual Analytics")

# Sales by Country
st.subheader("🌍 Sales by Country")
sales_by_country = (
    filtered_data.groupby("country")["totalamount"].sum().reset_index().sort_values("totalamount", ascending=False)
)
fig = px.bar(
    sales_by_country,
    x="country",
    y="totalamount",
    title="Total Revenue by Country",
    color="totalamount",
    color_continuous_scale=px.colors.sequential.Blues,
)
st.plotly_chart(fig, use_container_width=True)

# Top Selling Products
st.subheader("📦 Top Selling Products")
top_products = (
    filtered_data.groupby("description")["quantity"].sum().reset_index().sort_values("quantity", ascending=False).head(10)
)
fig = px.bar(
    top_products,
    x="quantity",
    y="description",
    orientation='h',
    title="Top Selling Products",
    color="quantity",
    color_continuous_scale=px.colors.sequential.Purples,
)
st.plotly_chart(fig, use_container_width=True)

# Sales Trend Over Time
st.subheader("📈 Sales Trend Over Time")
sales_over_time = (
    filtered_data.groupby(filtered_data['invoicedate'].dt.date)["totalamount"].sum().reset_index()
)
fig = px.line(
    sales_over_time,
    x="invoicedate",
    y="totalamount",
    title="Sales Trend by Revenue",
    markers=True,
    color_discrete_sequence=["#2A9D8F"],
)
st.plotly_chart(fig, use_container_width=True)

# Sales by Year
st.subheader("📅 Sales by Year")
sales_by_year = (
    filtered_data.groupby("year")["totalamount"].sum().reset_index().sort_values("year")
)
fig = px.bar(
    sales_by_year,
    x="year",
    y="totalamount",
    title="Total Revenue by Year",
    color="totalamount",
    color_continuous_scale=px.colors.sequential.Viridis,
)
st.plotly_chart(fig, use_container_width=True)

# ==================== ML INSIGHTS SECTION ====================
st.markdown("---")
st.markdown("## 🤖 Machine Learning Insights")

# ML Model 1: Customer Segmentation
st.markdown("### 👥 Customer Segmentation Analysis")
try:
    segments_query = """
    SELECT segment, 
           COUNT(*) as customer_count,
           AVG(totalpurchases) as avg_purchases,
           AVG(numberoftransactions) as avg_transactions
    FROM customer_segments
    GROUP BY segment
    ORDER BY customer_count DESC
    """
    segments_data = get_hive_data(segments_query)
    segments_data.columns = [col.split('.')[-1] for col in segments_data.columns]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Pie chart for segment distribution
        fig_seg = px.pie(
            segments_data,
            values='customer_count',
            names='segment',
            title='Customer Distribution by Segment',
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig_seg, use_container_width=True)
    
    with col2:
        # Bar chart for segment metrics
        fig_metrics = px.bar(
            segments_data,
            x='segment',
            y=['avg_purchases', 'avg_transactions'],
            title='Average Metrics by Segment',
            barmode='group',
            color_discrete_sequence=['#2A9D8F', '#E76F51']
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    st.dataframe(segments_data, use_container_width=True)
except Exception as e:
    st.warning("⚠️ Customer Segmentation data not available yet. Run the ML pipeline first.")
    st.code(str(e))

# ML Model: RFM Analysis
st.markdown("### 💎 RFM Analysis (Recency, Frequency, Monetary)")
try:
    rfm_query = """
    SELECT rfm_segment,
           churn_risk,
           COUNT(*) as customer_count,
           AVG(recency) as avg_recency_days,
           AVG(frequency) as avg_frequency,
           AVG(monetary) as avg_monetary,
           AVG(rfm_score) as avg_rfm_score
    FROM customer_rfm_analysis
    GROUP BY rfm_segment, churn_risk
    ORDER BY avg_rfm_score DESC
    """
    rfm_data = get_hive_data(rfm_query)
    rfm_data.columns = [col.split('.')[-1] for col in rfm_data.columns]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # RFM Segment Distribution
        segment_dist = rfm_data.groupby('rfm_segment')['customer_count'].sum().reset_index()
        fig_rfm_seg = px.pie(
            segment_dist,
            values='customer_count',
            names='rfm_segment',
            title='Customer Distribution by RFM Segment',
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig_rfm_seg, use_container_width=True)
    
    with col2:
        # Churn Risk Distribution
        risk_dist = rfm_data.groupby('churn_risk')['customer_count'].sum().reset_index()
        fig_risk = px.bar(
            risk_dist,
            x='churn_risk',
            y='customer_count',
            title='Churn Risk Distribution',
            color='churn_risk',
            color_discrete_map={'Low': '#2A9D8F', 'Medium': '#F4A261', 'High': '#E76F51'}
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col3:
        # RFM Score by Segment
        fig_rfm_score = px.bar(
            segment_dist,
            x='rfm_segment',
            y=rfm_data.groupby('rfm_segment')['avg_rfm_score'].mean().values,
            title='Average RFM Score by Segment',
            color_discrete_sequence=['#264653']
        )
        fig_rfm_score.update_xaxis(tickangle=-45)
        st.plotly_chart(fig_rfm_score, use_container_width=True)
    
    # Top Champions
    champions_query = """
    SELECT customerid, recency, frequency, monetary, rfm_score, rfm_segment
    FROM customer_rfm_analysis
    WHERE rfm_segment = 'Champions'
    ORDER BY rfm_score DESC
    LIMIT 10
    """
    champions_data = get_hive_data(champions_query)
    champions_data.columns = [col.split('.')[-1] for col in champions_data.columns]
    
    st.markdown("**🏆 Top 10 Champions (Most Valuable Customers):**")
    st.dataframe(champions_data, use_container_width=True)
    
    st.markdown("**📊 Detailed RFM Segment Metrics:**")
    st.dataframe(rfm_data, use_container_width=True)
except Exception as e:
    st.warning("⚠️ RFM Analysis data not available yet. Run the ML pipeline first.")
    st.code(str(e))

# ML Model 2: Sales Forecasting
st.markdown("### 📈 Sales Forecast (Next 3 Months)")
try:
    forecast_query = """
    SELECT month, year, predicted_sales
    FROM sales_forecast
    WHERE is_future = true
    ORDER BY year, month
    LIMIT 3
    """
    forecast_data = get_hive_data(forecast_query)
    forecast_data.columns = [col.split('.')[-1] for col in forecast_data.columns]
    
    # Historical sales
    historical_query = """
    SELECT month, year, totalsales as actual_sales
    FROM monthly_sales
    ORDER BY year DESC, month DESC
    LIMIT 12
    """
    historical_data = get_hive_data(historical_query)
    historical_data.columns = [col.split('.')[-1] for col in historical_data.columns]
    
    # Create date strings
    forecast_data['date'] = forecast_data['year'].astype(str) + '-' + forecast_data['month'].astype(str).str.zfill(2)
    historical_data['date'] = historical_data['year'].astype(str) + '-' + historical_data['month'].astype(str).str.zfill(2)
    
    # Combine for visualization
    historical_data['type'] = 'Actual'
    forecast_data['type'] = 'Forecast'
    historical_data['sales'] = historical_data['actual_sales']
    forecast_data['sales'] = forecast_data['predicted_sales']
    
    combined_data = pd.concat([
        historical_data[['date', 'sales', 'type']],
        forecast_data[['date', 'sales', 'type']]
    ])
    
    fig_forecast = px.line(
        combined_data,
        x='date',
        y='sales',
        color='type',
        title='Sales Forecast with Historical Data',
        markers=True,
        color_discrete_map={'Actual': '#2A9D8F', 'Forecast': '#E76F51'}
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.markdown("**Predicted Sales:**")
    st.dataframe(forecast_data[['month', 'year', 'predicted_sales']], use_container_width=True)
except Exception as e:
    st.warning("⚠️ Sales Forecast data not available yet. Run the ML pipeline first.")
    st.code(str(e))

# ML Model 3: Product Recommendations
st.markdown("### 🎯 Top Product Recommendations")
try:
    recommendations_query = """
    SELECT customerid, stockcode, prediction as recommendation_score
    FROM product_recommendations
    ORDER BY prediction DESC
    LIMIT 20
    """
    recommendations_data = get_hive_data(recommendations_query)
    recommendations_data.columns = [col.split('.')[-1] for col in recommendations_data.columns]
    
    st.dataframe(recommendations_data, use_container_width=True)
    
    # Show recommendation distribution
    rec_summary_query = """
    SELECT COUNT(DISTINCT customerid) as customers_with_recs,
           COUNT(*) as total_recommendations,
           AVG(prediction) as avg_score
    FROM product_recommendations
    """
    rec_summary = get_hive_data(rec_summary_query)
    rec_summary.columns = [col.split('.')[-1] for col in rec_summary.columns]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👥 Customers", f"{int(rec_summary['customers_with_recs'].values[0]):,}")
    with col2:
        st.metric("🎁 Total Recommendations", f"{int(rec_summary['total_recommendations'].values[0]):,}")
    with col3:
        st.metric("⭐ Avg Score", f"{rec_summary['avg_score'].values[0]:.2f}")
except Exception as e:
    st.warning("⚠️ Product Recommendations data not available yet. Run the ML pipeline first.")
    st.code(str(e))

# ML Model 4: Churn Prediction
st.markdown("### ⚠️ Customer Churn Risk Analysis")
try:
    churn_query = """
    SELECT churn_risk, 
           COUNT(*) as customer_count,
           AVG(churn_probability) as avg_probability
    FROM customer_churn_prediction
    GROUP BY churn_risk
    ORDER BY CASE 
        WHEN churn_risk = 'High' THEN 1
        WHEN churn_risk = 'Medium' THEN 2
        WHEN churn_risk = 'Low' THEN 3
        ELSE 4
    END
    """
    churn_data = get_hive_data(churn_query)
    churn_data.columns = [col.split('.')[-1] for col in churn_data.columns]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Bar chart for churn risk distribution
        fig_churn = px.bar(
            churn_data,
            x='churn_risk',
            y='customer_count',
            title='Customer Distribution by Churn Risk',
            color='churn_risk',
            color_discrete_map={'High': '#E63946', 'Medium': '#F77F00', 'Low': '#06D6A0'}
        )
        st.plotly_chart(fig_churn, use_container_width=True)
    
    with col2:
        # Gauge chart for high-risk customers
        high_risk_count = churn_data[churn_data['churn_risk'] == 'High']['customer_count'].values[0] if len(churn_data[churn_data['churn_risk'] == 'High']) > 0 else 0
        total_customers = churn_data['customer_count'].sum()
        high_risk_pct = (high_risk_count / total_customers * 100) if total_customers > 0 else 0
        
        fig_gauge = px.pie(
            churn_data,
            values='customer_count',
            names='churn_risk',
            title='Churn Risk Distribution',
            hole=0.4,
            color='churn_risk',
            color_discrete_map={'High': '#E63946', 'Medium': '#F77F00', 'Low': '#06D6A0'}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.dataframe(churn_data, use_container_width=True)
    
    st.markdown(f"**⚠️ High Risk Alert:** {int(high_risk_count):,} customers ({high_risk_pct:.1f}%) are at high risk of churning!")
except Exception as e:
    st.warning("⚠️ Churn Prediction data not available yet. Run the ML pipeline first.")
    st.code(str(e))

# ML Model 5: Anomaly Detection
st.markdown("### 🚨 Transaction Anomalies")
try:
    anomaly_query = """
    SELECT invoiceno, invoicedate, customerid, quantity, unitprice, totalamount,
           anomaly_score, severity
    FROM transaction_anomalies
    WHERE severity IN ('Critical', 'High')
    ORDER BY anomaly_score DESC
    LIMIT 20
    """
    anomaly_data = get_hive_data(anomaly_query)
    anomaly_data.columns = [col.split('.')[-1] for col in anomaly_data.columns]
    
    # Summary metrics
    anomaly_summary_query = """
    SELECT severity,
           COUNT(*) as anomaly_count,
           AVG(anomaly_score) as avg_score,
           SUM(totalamount) as total_amount
    FROM transaction_anomalies
    GROUP BY severity
    ORDER BY CASE 
        WHEN severity = 'Critical' THEN 1
        WHEN severity = 'High' THEN 2
        WHEN severity = 'Medium' THEN 3
        ELSE 4
    END
    """
    anomaly_summary = get_hive_data(anomaly_summary_query)
    anomaly_summary.columns = [col.split('.')[-1] for col in anomaly_summary.columns]
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    critical_count = anomaly_summary[anomaly_summary['severity'] == 'Critical']['anomaly_count'].values[0] if len(anomaly_summary[anomaly_summary['severity'] == 'Critical']) > 0 else 0
    high_count = anomaly_summary[anomaly_summary['severity'] == 'High']['anomaly_count'].values[0] if len(anomaly_summary[anomaly_summary['severity'] == 'High']) > 0 else 0
    
    with col1:
        st.metric("🔴 Critical Anomalies", f"{int(critical_count):,}")
    with col2:
        st.metric("🟠 High Anomalies", f"{int(high_count):,}")
    with col3:
        total_anomalies = anomaly_summary['anomaly_count'].sum()
        st.metric("📊 Total Detected", f"{int(total_anomalies):,}")
    
    # Bar chart for anomaly distribution
    fig_anomaly = px.bar(
        anomaly_summary,
        x='severity',
        y='anomaly_count',
        title='Anomaly Distribution by Severity',
        color='severity',
        color_discrete_map={'Critical': '#DC2F02', 'High': '#E85D04', 'Medium': '#F48C06', 'Low': '#FAA307'}
    )
    st.plotly_chart(fig_anomaly, use_container_width=True)
    
    st.markdown("**Top Critical/High Anomalies:**")
    st.dataframe(anomaly_data, use_container_width=True)
except Exception as e:
    st.warning("⚠️ Anomaly Detection data not available yet. Run the ML pipeline first.")
    st.code(str(e))

st.markdown("---")
st.markdown("*Dashboard powered by Streamlit, Hive, and PySpark MLlib* 🚀")