import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Page configuration
st.set_page_config(
    page_title="E-Commerce ML Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
ML_RESULTS_DIR = './ml_results'
DATA_FILE = './data/data.csv'

# Custom CSS
st.markdown("""
    <style>
        .main {background-color: #f5f7fa;}
        .stMetric {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
        h1 {color: #1e3a8a;}
        h2 {color: #2563eb;}
        h3 {color: #3b82f6;}
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🛒 E-Commerce ML Analytics Dashboard")
st.markdown("**Powered by Direct CSV Analysis** | No complex pipeline needed!")

# Sidebar
with st.sidebar:
    st.header("📊 Dashboard Controls")
    st.info("💡 **Tip**: Run `python ml_analysis/run_all_analyses.py` to update ML insights")
    
    # Check if ML results exist
    results_exist = os.path.exists(ML_RESULTS_DIR)
    if results_exist:
        files = os.listdir(ML_RESULTS_DIR)
        st.success(f"✅ {len(files)} ML result files found")
        st.markdown("**Available Analyses:**")
        for f in files:
            st.text(f"  • {f}")
    else:
        st.warning("⚠️ No ML results found. Run analyses first!")

st.markdown("---")

# === OVERVIEW METRICS ===
st.header("📈 Overview Metrics")

try:
    # Load raw data for overview
    df = pd.read_csv(DATA_FILE, encoding='latin-1')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    df_clean = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0) & (df['TotalAmount'] > 0)]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Revenue", f"${df_clean['TotalAmount'].sum():,.0f}")
    with col2:
        st.metric("Total Orders", f"{df_clean['InvoiceNo'].nunique():,}")
    with col3:
        st.metric("Unique Customers", f"{df_clean['CustomerID'].nunique():,}")
    with col4:
        st.metric("Unique Products", f"{df_clean['Description'].nunique():,}")
        
except Exception as e:
    st.error(f"Error loading overview data: {e}")

st.markdown("---")

# === RFM ANALYSIS ===
st.header("💎 RFM Analysis")

rfm_file = os.path.join(ML_RESULTS_DIR, 'rfm_analysis.csv')
if os.path.exists(rfm_file):
    try:
        rfm_data = pd.read_csv(rfm_file)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Segment distribution
            segment_dist = rfm_data['RFM_Segment'].value_counts().reset_index()
            segment_dist.columns = ['Segment', 'Count']
            
            fig_seg = px.pie(
                segment_dist,
                values='Count',
                names='Segment',
                title='Customer Segments',
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            st.plotly_chart(fig_seg, use_container_width=True)
        
        with col2:
            # Churn risk
            risk_dist = rfm_data['Churn_Risk'].value_counts().reset_index()
            risk_dist.columns = ['Risk', 'Count']
            
            fig_risk = px.bar(
                risk_dist,
                x='Risk',
                y='Count',
                title='Churn Risk Distribution',
                color='Risk',
                color_discrete_map={'Low': '#2A9D8F', 'Medium': '#F4A261', 'High': '#E76F51'}
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        
        with col3:
            # RFM Score distribution
            fig_score = px.histogram(
                rfm_data,
                x='RFM_Score',
                title='RFM Score Distribution',
                nbins=20,
                color_discrete_sequence=['#264653']
            )
            st.plotly_chart(fig_score, use_container_width=True)
        
        # Top Champions
        st.subheader("🏆 Top 10 Champions")
        champions = rfm_data[rfm_data['RFM_Segment'] == 'Champions'].nlargest(10, 'RFM_Score')
        st.dataframe(
            champions[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'RFM_Score']],
            use_container_width=True,
            hide_index=True
        )
        
        # Segment metrics
        st.subheader("📊 Segment Metrics")
        segment_metrics = rfm_data.groupby('RFM_Segment').agg({
            'CustomerID': 'count',
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'RFM_Score': 'mean'
        }).round(2)
        segment_metrics.columns = ['Customers', 'Avg Recency', 'Avg Frequency', 'Avg Monetary', 'Avg Score']
        st.dataframe(segment_metrics, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading RFM data: {e}")
else:
    st.warning("⚠️ RFM analysis not available. Run `run_all_analyses.py` first.")

st.markdown("---")

# === CUSTOMER SEGMENTATION ===
st.header("👥 Customer Segmentation")

seg_file = os.path.join(ML_RESULTS_DIR, 'customer_segments.csv')
if os.path.exists(seg_file):
    try:
        seg_data = pd.read_csv(seg_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Segment distribution
            segment_dist = seg_data['Segment'].value_counts().reset_index()
            segment_dist.columns = ['Segment', 'Count']
            
            fig_seg = px.pie(
                segment_dist,
                values='Count',
                names='Segment',
                title='Customer Segment Distribution',
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig_seg, use_container_width=True)
        
        with col2:
            # Segment value
            segment_value = seg_data.groupby('Segment')['TotalSpending'].sum().reset_index()
            
            fig_value = px.bar(
                segment_value,
                x='Segment',
                y='TotalSpending',
                title='Total Revenue by Segment',
                color_discrete_sequence=['#2A9D8F']
            )
            fig_value.update_xaxis(tickangle=-45)
            st.plotly_chart(fig_value, use_container_width=True)
        
        # Segment metrics
        st.subheader("📊 Segment Metrics")
        metrics = seg_data.groupby('Segment').agg({
            'CustomerID': 'count',
            'TotalSpending': 'mean',
            'TotalOrders': 'mean',
            'AvgOrderValue': 'mean'
        }).round(2)
        metrics.columns = ['Customers', 'Avg Spending', 'Avg Orders', 'Avg Order Value']
        st.dataframe(metrics, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading segmentation data: {e}")
else:
    st.warning("⚠️ Customer segmentation not available. Run `run_all_analyses.py` first.")

st.markdown("---")

# === MARKET BASKET ANALYSIS ===
st.header("🛒 Market Basket Analysis")

basket_file = os.path.join(ML_RESULTS_DIR, 'product_associations.csv')
if os.path.exists(basket_file):
    try:
        basket_data = pd.read_csv(basket_file)
        
        st.subheader("🔗 Top Product Associations")
        st.markdown("**Products frequently bought together:**")
        
        # Top 20 associations
        top_20 = basket_data.head(20)
        
        # Create a nice display
        for idx, row in top_20.iterrows():
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.text(f"📦 {row['Product_A'][:50]}")
            with col2:
                st.text(f"📦 {row['Product_B'][:50]}")
            with col3:
                st.metric("Support", f"{row['Support']}%", f"{row['Frequency']} times")
        
        # Download option
        st.download_button(
            label="📥 Download Full Results",
            data=basket_data.to_csv(index=False),
            file_name="product_associations.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error loading market basket data: {e}")
else:
    st.warning("⚠️ Market basket analysis not available. Run `run_all_analyses.py` first.")

st.markdown("---")

# === SALES FORECAST ===
st.header("📈 Sales Forecasting")

forecast_file = os.path.join(ML_RESULTS_DIR, 'sales_forecast.csv')
if os.path.exists(forecast_file):
    try:
        forecast_data = pd.read_csv(forecast_file)
        
        # Historical vs Predicted
        fig_forecast = go.Figure()
        
        historical = forecast_data[forecast_data['Type'] == 'Historical']
        future = forecast_data[forecast_data['Type'] == 'Forecast']
        
        fig_forecast.add_trace(go.Scatter(
            x=historical['YearMonth'],
            y=historical['Revenue'],
            mode='lines+markers',
            name='Actual Sales',
            line=dict(color='#2A9D8F', width=3)
        ))
        
        fig_forecast.add_trace(go.Scatter(
            x=historical['YearMonth'],
            y=historical['Predicted_Revenue'],
            mode='lines',
            name='Model Prediction',
            line=dict(color='#E76F51', width=2, dash='dash')
        ))
        
        fig_forecast.add_trace(go.Scatter(
            x=future['YearMonth'],
            y=future['Predicted_Revenue'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#F4A261', width=3),
            marker=dict(size=10)
        ))
        
        fig_forecast.update_layout(
            title='Sales Forecast',
            xaxis_title='Month',
            yaxis_title='Revenue ($)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast table
        st.subheader("🔮 3-Month Forecast")
        st.dataframe(
            future[['YearMonth', 'Predicted_Revenue']].rename(columns={'YearMonth': 'Month', 'Predicted_Revenue': 'Predicted Revenue'}),
            use_container_width=True,
            hide_index=True
        )
        
    except Exception as e:
        st.error(f"Error loading forecast data: {e}")
else:
    st.warning("⚠️ Sales forecast not available. Run `run_all_analyses.py` first.")

# Footer
st.markdown("---")
st.markdown("**💡 How to Update:** Run `python ml_analysis/run_all_analyses.py` to refresh all ML insights")
