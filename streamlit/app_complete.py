import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import os

# Configure Plotly default template for dark theme
pio.templates["custom_dark"] = pio.templates["plotly_dark"]
pio.templates["custom_dark"].layout.update({
    'plot_bgcolor': '#2d2d2d',
    'paper_bgcolor': '#1a1a1a',
    'font': {'color': '#e0e0e0', 'family': 'Arial, sans-serif', 'size': 12},
    'title': {'font': {'color': '#90cdf4', 'size': 16, 'family': 'Arial, sans-serif'}},
    'xaxis': {
        'gridcolor': '#404040', 
        'color': '#e0e0e0', 
        'showgrid': True,
        'title': {'font': {'color': '#e0e0e0', 'size': 14}},
        'tickfont': {'color': '#e0e0e0', 'size': 12}
    },
    'yaxis': {
        'gridcolor': '#404040', 
        'color': '#e0e0e0', 
        'showgrid': True,
        'title': {'font': {'color': '#e0e0e0', 'size': 14}},
        'tickfont': {'color': '#e0e0e0', 'size': 12}
    },
    'legend': {
        'bgcolor': 'rgba(45,45,45,0.9)', 
        'bordercolor': '#606060', 
        'borderwidth': 1,
        'font': {'color': '#e0e0e0', 'size': 12}
    },
    'hoverlabel': {
        'bgcolor': '#404040',
        'font': {'color': '#ffffff', 'family': 'Arial, sans-serif', 'size': 13},
        'bordercolor': '#606060'
    }
})
pio.templates.default = "custom_dark"

# Page configuration
st.set_page_config(
    page_title="E-Commerce Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
ML_RESULTS_DIR = os.path.join(parent_dir, 'ml_results')
DATA_FILE = os.path.join(parent_dir, 'data', 'data.csv')

# Chart styling for visibility in all modes
CHART_TEMPLATE = {
    'plot_bgcolor': 'white',
    'paper_bgcolor': 'white',
    'font': {'color': '#2d3748', 'family': 'Arial, sans-serif'},
    'title': {'font': {'color': '#1a365d', 'size': 16, 'family': 'Arial, sans-serif'}},
    'xaxis': {'gridcolor': '#e2e8f0', 'color': '#2d3748'},
    'yaxis': {'gridcolor': '#e2e8f0', 'color': '#2d3748'},
    'legend': {'bgcolor': 'rgba(255,255,255,0.9)', 'bordercolor': '#e2e8f0', 'borderwidth': 1}
}

def style_chart(fig):
    """Apply consistent dark theme styling to charts"""
    fig.update_layout(
        plot_bgcolor='#2d2d2d',
        paper_bgcolor='#1a1a1a',
        font=dict(color='#e0e0e0', family='Arial, sans-serif', size=12),
        title_font=dict(color='#90cdf4', size=16),
        xaxis=dict(
            gridcolor='#404040', 
            color='#e0e0e0',
            title_font=dict(color='#e0e0e0', size=14),
            tickfont=dict(color='#e0e0e0', size=12)
        ),
        yaxis=dict(
            gridcolor='#404040', 
            color='#e0e0e0',
            title_font=dict(color='#e0e0e0', size=14),
            tickfont=dict(color='#e0e0e0', size=12)
        ),
        legend=dict(
            bgcolor='rgba(45,45,45,0.9)', 
            bordercolor='#606060', 
            borderwidth=1,
            font=dict(color='#e0e0e0', size=12)
        ),
        hoverlabel=dict(
            bgcolor='#404040',
            font_size=13,
            font_family='Arial, sans-serif',
            font_color='#ffffff',
            bordercolor='#606060'
        )
    )
    # Update traces to have visible hover text
    fig.update_traces(
        hoverlabel=dict(
            bgcolor='#404040',
            font=dict(color='#ffffff', size=13)
        )
    )
    return fig

# Custom CSS for dark theme matching
st.markdown("""
    <style>
        /* Dark theme for main content */
        .main {
            background-color: #1a1a1a !important;
        }
        
        /* Metric cards - dark theme */
        .stMetric {
            background-color: #2d2d2d !important;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            border: 1px solid #404040;
        }
        
        .stMetric label {
            color: #b0b0b0 !important;
            font-weight: 600;
        }
        
        .stMetric [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-weight: 700;
        }
        
        .stMetric [data-testid="stMetricDelta"] {
            color: #90cdf4 !important;
        }
        
        /* Headers - light text on dark */
        h1, h2, h3, h4, h5, h6 {
            color: #90cdf4 !important;
            font-weight: 600;
        }
        
        h1 {
            font-weight: 700;
            border-bottom: 3px solid #3182ce;
            padding-bottom: 10px;
        }
        
        /* All text elements - light on dark */
        p, span, div, label, .stMarkdown {
            color: #e0e0e0 !important;
        }
        
        /* Data tables - dark theme */
        .dataframe {
            background-color: #2d2d2d !important;
            color: #e0e0e0 !important;
        }
        
        .dataframe th {
            background-color: #404040 !important;
            color: #ffffff !important;
            font-weight: 600;
        }
        
        .dataframe td {
            color: #e0e0e0 !important;
            border-color: #404040 !important;
        }
        
        /* Tabs - dark theme */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #2d2d2d !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #b0b0b0 !important;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            color: #90cdf4 !important;
            font-weight: 700;
            background-color: #404040 !important;
        }
        
        /* Insight boxes - dark theme */
        .insight-box {
            background-color: #1e3a5f !important;
            border-left: 4px solid #3182ce;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            color: #e0e0e0 !important;
        }
        
        .warning-box {
            background-color: #3d1f1f !important;
            border-left: 4px solid #fc8181;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            color: #ffcccc !important;
        }
        
        /* Sidebar - darker */
        section[data-testid="stSidebar"] {
            background-color: #262626 !important;
        }
        
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #90cdf4 !important;
        }
        
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] label {
            color: #d0d0d0 !important;
        }
        
        section[data-testid="stSidebar"] .stRadio label {
            color: #d0d0d0 !important;
        }
        
        /* Buttons */
        .stButton button {
            background-color: #3182ce !important;
            color: #ffffff !important;
            border: none;
            font-weight: 500;
        }
        
        .stButton button:hover {
            background-color: #2c5282 !important;
        }
        
        /* Download buttons */
        .stDownloadButton button {
            background-color: #48bb78 !important;
            color: #ffffff !important;
        }
        
        /* Selectbox and multiselect */
        .stSelectbox label,
        .stMultiSelect label,
        .stSlider label,
        .stDateInput label {
            color: #e0e0e0 !important;
            font-weight: 600;
        }
        
        /* Input fields */
        .stSelectbox div[data-baseweb="select"] {
            background-color: #2d2d2d !important;
        }
        
        .stMultiSelect div[data-baseweb="select"] {
            background-color: #2d2d2d !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #2d2d2d !important;
            color: #e0e0e0 !important;
            font-weight: 600;
        }
        
        /* Info, warning, success boxes */
        .stAlert {
            background-color: #2d2d2d !important;
            color: #e0e0e0 !important;
        }
        
        /* Override any light text on dark backgrounds */
        [data-testid="stMarkdownContainer"] {
            color: #e0e0e0 !important;
        }
        
        /* Ensure table headers are visible */
        thead tr th {
            background-color: #404040 !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            border-color: #505050 !important;
        }
        
        tbody tr td {
            color: #e0e0e0 !important;
            border-color: #404040 !important;
        }
        
        /* Date input */
        .stDateInput input {
            background-color: #2d2d2d !important;
            color: #e0e0e0 !important;
        }
        
        /* Slider */
        .stSlider {
            color: #e0e0e0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("📊 E-Commerce Analytics Platform")

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    
    st.markdown("---")
    
    # System Status - Minimal
    results_exist = os.path.exists(ML_RESULTS_DIR)
    if results_exist:
        files = [f for f in os.listdir(ML_RESULTS_DIR) if f.endswith('.csv')]
        st.markdown(f"**System Status:** ✅ Active")
        st.markdown(f"**Models:** {len(files)} | **Last Update:** Today")
    else:
        st.markdown("**System Status:** ⚠️ Pending")
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📖 Documentation")
    st.markdown("""
    - [Quick Start Guide](QUICK_START.md)
    - [ML Approach](ML_DIRECT_APPROACH_SUCCESS.md)
    - [Setup Guide](SETUP_GUIDE.md)
    """)

st.markdown("---")

# ================== SECTION 1: BUSINESS OVERVIEW ==================
st.header("📈 Business Overview")

try:
    # Load and process data
    df = pd.read_csv(DATA_FILE, encoding='latin-1')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    df_clean = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0) & (df['TotalAmount'] > 0) & df['CustomerID'].notna()]
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_revenue = df_clean['TotalAmount'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.0f}")
    
    with col2:
        total_orders = df_clean['InvoiceNo'].nunique()
        st.metric("Total Orders", f"{total_orders:,}")
    
    with col3:
        total_customers = df_clean['CustomerID'].nunique()
        st.metric("Unique Customers", f"{total_customers:,}")
    
    with col4:
        total_products = df_clean['Description'].nunique()
        st.metric("Unique Products", f"{total_products:,}")
    
    with col5:
        avg_order_value = total_revenue / total_orders
        st.metric("Avg Order Value", f"${avg_order_value:.2f}")
    
    st.markdown("---")
    
    # Date Range Filter
    st.subheader("🔍 Data Filters")
    col_filter1, col_filter2, col_filter3 = st.columns([2, 2, 1])
    
    with col_filter1:
        min_date = df_clean['InvoiceDate'].min().date()
        max_date = df_clean['InvoiceDate'].max().date()
        date_range = st.date_input(
            "Select Date Range:",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
    
    with col_filter2:
        all_countries = sorted(df_clean['Country'].unique().tolist())
        selected_countries = st.multiselect(
            "Select Countries:",
            all_countries,
            default=all_countries[:5]
        )
    
    with col_filter3:
        top_n = st.slider("Top N Results:", 5, 50, 10)
    
    # Apply filters
    if len(date_range) == 2:
        filtered_df = df_clean[
            (df_clean['InvoiceDate'].dt.date >= date_range[0]) &
            (df_clean['InvoiceDate'].dt.date <= date_range[1])
        ]
    else:
        filtered_df = df_clean
    
    if selected_countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]
    
    st.markdown("---")
    
    # Visualizations in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Sales Analytics", "🌍 Geographic Analysis", "📦 Product Insights", "📅 Temporal Trends"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue trend over time
            st.subheader("Revenue Trend Over Time")
            daily_sales = filtered_df.groupby(filtered_df['InvoiceDate'].dt.date)['TotalAmount'].sum().reset_index()
            daily_sales.columns = ['Date', 'Revenue']
            
            fig_trend = px.line(
                daily_sales,
                x='Date',
                y='Revenue',
                title='Daily Revenue',
                markers=True
            )
            fig_trend.update_traces(line_color='#3182ce', line_width=2)
            fig_trend = style_chart(fig_trend)
            fig_trend.update_layout(hovermode='x unified')
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            # Monthly revenue comparison
            st.subheader("Monthly Revenue Comparison")
            filtered_df['YearMonth'] = filtered_df['InvoiceDate'].dt.to_period('M').astype(str)
            monthly_sales = filtered_df.groupby('YearMonth')['TotalAmount'].sum().reset_index()
            
            fig_monthly = px.bar(
                monthly_sales,
                x='YearMonth',
                y='TotalAmount',
                title='Monthly Revenue',
                color='TotalAmount',
                color_continuous_scale='Blues'
            )
            fig_monthly = style_chart(fig_monthly)
            fig_monthly.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_monthly, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales by country
            st.subheader(f"Top {top_n} Countries by Revenue")
            country_sales = filtered_df.groupby('Country')['TotalAmount'].sum().sort_values(ascending=False).head(top_n).reset_index()
            
            fig_country = px.bar(
                country_sales,
                x='TotalAmount',
                y='Country',
                orientation='h',
                title='Revenue by Country',
                color='TotalAmount',
                color_continuous_scale='Viridis'
            )
            fig_country = style_chart(fig_country)
            st.plotly_chart(fig_country, use_container_width=True)
        
        with col2:
            # Customer distribution by country
            st.subheader(f"Top {top_n} Countries by Customers")
            country_customers = filtered_df.groupby('Country')['CustomerID'].nunique().sort_values(ascending=False).head(top_n).reset_index()
            country_customers.columns = ['Country', 'Customers']
            
            fig_cust_country = px.pie(
                country_customers,
                values='Customers',
                names='Country',
                title='Customer Distribution',
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig_cust_country = style_chart(fig_cust_country)
            st.plotly_chart(fig_cust_country, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top selling products by quantity
            st.subheader(f"Top {top_n} Products by Quantity")
            product_qty = filtered_df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(top_n).reset_index()
            
            fig_prod_qty = px.bar(
                product_qty,
                x='Quantity',
                y='Description',
                orientation='h',
                title='Products by Quantity Sold',
                color='Quantity',
                color_continuous_scale='Purples'
            )
            fig_prod_qty = style_chart(fig_prod_qty)
            st.plotly_chart(fig_prod_qty, use_container_width=True)
        
        with col2:
            # Top products by revenue
            st.subheader(f"Top {top_n} Products by Revenue")
            product_rev = filtered_df.groupby('Description')['TotalAmount'].sum().sort_values(ascending=False).head(top_n).reset_index()
            
            fig_prod_rev = px.bar(
                product_rev,
                x='TotalAmount',
                y='Description',
                orientation='h',
                title='Products by Revenue',
                color='TotalAmount',
                color_continuous_scale='Greens'
            )
            fig_prod_rev = style_chart(fig_prod_rev)
            st.plotly_chart(fig_prod_rev, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales by day of week
            st.subheader("Sales by Day of Week")
            filtered_df['DayOfWeek'] = filtered_df['InvoiceDate'].dt.day_name()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_sales = filtered_df.groupby('DayOfWeek')['TotalAmount'].sum().reindex(day_order).reset_index()
            
            fig_day = px.bar(
                day_sales,
                x='DayOfWeek',
                y='TotalAmount',
                title='Revenue by Day of Week',
                color='TotalAmount',
                color_continuous_scale='Teal'
            )
            fig_day = style_chart(fig_day)
            st.plotly_chart(fig_day, use_container_width=True)
        
        with col2:
            # Sales by hour
            st.subheader("Sales by Hour of Day")
            filtered_df['Hour'] = filtered_df['InvoiceDate'].dt.hour
            hour_sales = filtered_df.groupby('Hour')['TotalAmount'].sum().reset_index()
            
            fig_hour = px.line(
                hour_sales,
                x='Hour',
                y='TotalAmount',
                title='Revenue by Hour',
                markers=True
            )
            fig_hour.update_traces(line_color='#e53e3e', line_width=3)
            fig_hour = style_chart(fig_hour)
            st.plotly_chart(fig_hour, use_container_width=True)
    
    # Data export
    st.markdown("---")
    st.subheader("📥 Data Export")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"Filtered dataset contains {len(filtered_df):,} transactions")
    with col2:
        st.download_button(
            label="Download CSV",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name='filtered_ecommerce_data.csv',
            mime='text/csv',
        )

except Exception as e:
    st.error(f"Error loading business data: {str(e)}")

st.markdown("---")

# ================== SECTION 2: MACHINE LEARNING INSIGHTS ==================
st.header("🤖 Machine Learning Insights")
st.markdown("**Advanced analytics powered by predictive models and clustering algorithms**")

# RFM Analysis
st.subheader("💎 RFM Analysis: Customer Value Segmentation")
st.markdown("**Model Target:** Segments customers by Recency (last purchase), Frequency (purchase count), and Monetary value (total spent) to identify high-value customers and churn risk.")
rfm_file = os.path.join(ML_RESULTS_DIR, 'rfm_analysis.csv')

if os.path.exists(rfm_file):
    try:
        rfm_data = pd.read_csv(rfm_file)
        
        # Key insights
        total_customers = len(rfm_data)
        champions = len(rfm_data[rfm_data['RFM_Segment'] == 'Champions'])
        at_risk = len(rfm_data[rfm_data['RFM_Segment'] == 'At Risk'])
        lost = len(rfm_data[rfm_data['RFM_Segment'] == 'Lost'])
        high_churn = len(rfm_data[rfm_data['Churn_Risk'] == 'High'])
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Champions", f"{champions:,}", f"{champions/total_customers*100:.1f}%")
        col3.metric("At Risk", f"{at_risk:,}", f"{at_risk/total_customers*100:.1f}%")
        col4.metric("Lost Customers", f"{lost:,}", f"{lost/total_customers*100:.1f}%")
        col5.metric("High Churn Risk", f"{high_churn:,}", f"{high_churn/total_customers*100:.1f}%")
        
        # Visualizations
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Segment distribution
            segment_dist = rfm_data['RFM_Segment'].value_counts().reset_index()
            segment_dist.columns = ['Segment', 'Count']
            
            fig_seg = px.pie(
                segment_dist,
                values='Count',
                names='Segment',
                title='Customer Segment Distribution',
                color_discrete_sequence=px.colors.sequential.Plasma,
                hole=0.3
            )
            fig_seg = style_chart(fig_seg)
            st.plotly_chart(fig_seg, use_container_width=True)
        
        with col2:
            # Churn risk
            risk_dist = rfm_data['Churn_Risk'].value_counts().reset_index()
            risk_dist.columns = ['Risk Level', 'Customer Count']
            
            fig_risk = px.bar(
                risk_dist,
                x='Risk Level',
                y='Customer Count',
                title='Churn Risk Distribution',
                color='Risk Level',
                color_discrete_map={'Low': '#48bb78', 'Medium': '#ed8936', 'High': '#f56565'}
            )
            fig_risk = style_chart(fig_risk)
            st.plotly_chart(fig_risk, use_container_width=True)
        
        with col3:
            # RFM Score distribution
            fig_score = px.histogram(
                rfm_data,
                x='RFM_Score',
                title='RFM Score Distribution',
                nbins=30,
                color_discrete_sequence=['#4299e1']
            )
            fig_score.update_layout(showlegend=False)
            fig_score = style_chart(fig_score)
            st.plotly_chart(fig_score, use_container_width=True)
        
        # Detailed segment analysis
        with st.expander("📊 Detailed Segment Metrics"):
            segment_metrics = rfm_data.groupby('RFM_Segment').agg({
                'CustomerID': 'count',
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean',
                'RFM_Score': 'mean'
            }).round(2)
            segment_metrics.columns = ['Customer Count', 'Avg Recency (Days)', 'Avg Frequency', 'Avg Monetary ($)', 'Avg RFM Score']
            segment_metrics = segment_metrics.sort_values('Avg RFM Score', ascending=False)
            st.dataframe(segment_metrics, use_container_width=True)
        
        # Champions list
        with st.expander("🏆 Top 20 Champions (Most Valuable Customers)"):
            champions_df = rfm_data[rfm_data['RFM_Segment'] == 'Champions'].nlargest(20, 'RFM_Score')
            display_cols = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'RFM_Score']
            st.dataframe(champions_df[display_cols], use_container_width=True, hide_index=True)
        
        # Key insights
        st.markdown("""
        <div class="insight-box">
        <strong>Key Insights:</strong><br>
        • Champions represent your most valuable customers - focus retention efforts here<br>
        • At-Risk customers need immediate engagement to prevent churn<br>
        • Lost customers may be recoverable with targeted win-back campaigns<br>
        • High churn risk customers require personalized attention and incentives
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error loading RFM data: {str(e)}")
else:
    st.warning("RFM analysis not available. Run ML analysis to generate insights.")

st.markdown("---")

# Customer Segmentation
st.subheader("👥 Customer Segmentation: Behavioral Clustering")
st.markdown("**Model Target:** Groups customers into behavioral clusters based on spending patterns and purchase behavior to enable personalized marketing strategies.")

seg_file = os.path.join(ML_RESULTS_DIR, 'customer_segments.csv')
if os.path.exists(seg_file):
    try:
        seg_data = pd.read_csv(seg_file)
        
        # Metrics
        segments = seg_data['Segment'].nunique()
        total_spending = seg_data['TotalSpending'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Segments", f"{segments}")
        col2.metric("Total Customer Spend", f"${total_spending:,.0f}")
        col3.metric("Avg Spending per Customer", f"${seg_data['TotalSpending'].mean():.2f}")
        col4.metric("Avg Orders per Customer", f"{seg_data['TotalOrders'].mean():.1f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Segment distribution
            segment_dist = seg_data['Segment'].value_counts().reset_index()
            segment_dist.columns = ['Segment', 'Customer Count']
            
            fig_seg = px.bar(
                segment_dist,
                x='Segment',
                y='Customer Count',
                title='Customer Count by Segment',
                color='Customer Count',
                color_continuous_scale='Teal'
            )
            fig_seg = style_chart(fig_seg)
            st.plotly_chart(fig_seg, use_container_width=True)
        
        with col2:
            # Revenue by segment
            segment_revenue = seg_data.groupby('Segment')['TotalSpending'].sum().reset_index()
            segment_revenue.columns = ['Segment', 'Total Revenue']
            
            fig_revenue = px.pie(
                segment_revenue,
                values='Total Revenue',
                names='Segment',
                title='Revenue Contribution by Segment',
                color_discrete_sequence=px.colors.sequential.RdBu,
                hole=0.4
            )
            fig_revenue = style_chart(fig_revenue)
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Segment characteristics
        col1, col2 = st.columns(2)
        
        with col1:
            # Spending patterns
            fig_spending = px.box(
                seg_data,
                x='Segment',
                y='TotalSpending',
                title='Spending Distribution by Segment',
                color='Segment'
            )
            fig_spending.update_layout(showlegend=False)
            fig_spending = style_chart(fig_spending)
            st.plotly_chart(fig_spending, use_container_width=True)
        
        with col2:
            # Order patterns
            fig_orders = px.scatter(
                seg_data,
                x='TotalOrders',
                y='TotalSpending',
                color='Segment',
                title='Orders vs Spending by Segment',
                size='AvgOrderValue',
                hover_data=['CustomerID']
            )
            fig_orders = style_chart(fig_orders)
            st.plotly_chart(fig_orders, use_container_width=True)
        
        # Detailed metrics
        with st.expander("📊 Detailed Segment Metrics"):
            metrics = seg_data.groupby('Segment').agg({
                'CustomerID': 'count',
                'TotalSpending': ['sum', 'mean'],
                'TotalOrders': 'mean',
                'AvgOrderValue': 'mean'
            }).round(2)
            metrics.columns = ['Customers', 'Total Revenue', 'Avg Spending', 'Avg Orders', 'Avg Order Value']
            st.dataframe(metrics, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>Segmentation Insights:</strong><br>
        • High-value segments contribute disproportionately to total revenue<br>
        • Different segments require tailored marketing and retention strategies<br>
        • Average order value varies significantly across segments<br>
        • Use segment characteristics to personalize customer experiences
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error loading segmentation data: {str(e)}")
else:
    st.warning("Customer segmentation not available. Run ML analysis to generate insights.")

st.markdown("---")

# Market Basket Analysis
st.subheader("🛒 Market Basket Analysis: Product Associations")
st.markdown("**Model Target:** Identifies products frequently purchased together to enable cross-selling, product bundling, and optimized store layout strategies.")

basket_file = os.path.join(ML_RESULTS_DIR, 'product_associations.csv')
if os.path.exists(basket_file):
    try:
        basket_data = pd.read_csv(basket_file)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Product Pairs Analyzed", f"{len(basket_data):,}")
        col2.metric("Avg Association Frequency", f"{basket_data['Frequency'].mean():.1f}")
        col3.metric("Max Support %", f"{basket_data['Support'].max():.2f}%")
        
        # Display top associations
        st.markdown(f"**Top {min(20, len(basket_data))} Product Associations:**")
        
        # Create a more visual display
        top_associations = basket_data.head(20)
        
        # Bar chart of top associations
        top_associations['Pair'] = top_associations['Product_A'].str[:30] + ' + ' + top_associations['Product_B'].str[:30]
        
        fig_basket = px.bar(
            top_associations,
            x='Frequency',
            y='Pair',
            orientation='h',
            title='Top Product Pairs by Co-occurrence Frequency',
            color='Support',
            color_continuous_scale='Viridis',
            hover_data=['Support']
        )
        fig_basket.update_layout(height=600)
        fig_basket = style_chart(fig_basket)
        st.plotly_chart(fig_basket, use_container_width=True)
        
        # Detailed table
        with st.expander("📋 View Complete Association Rules"):
            display_basket = basket_data.copy()
            display_basket['Support'] = display_basket['Support'].round(3)
            st.dataframe(display_basket, use_container_width=True, hide_index=True)
            
            # Download option
            st.download_button(
                label="Download Full Results (CSV)",
                data=basket_data.to_csv(index=False).encode('utf-8'),
                file_name="product_associations.csv",
                mime="text/csv"
            )
        
        st.markdown("""
        <div class="insight-box">
        <strong>Market Basket Insights:</strong><br>
        • Use product associations for cross-selling and bundling strategies<br>
        • Place frequently bought-together items near each other<br>
        • Create promotional bundles based on association patterns<br>
        • Optimize recommendation engines with these associations
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error loading market basket data: {str(e)}")
else:
    st.warning("Market basket analysis not available. Run ML analysis to generate insights.")

st.markdown("---")

# Sales Forecasting
st.subheader("📈 Sales Forecasting: Predictive Revenue Analysis")
st.markdown("**Model Target:** Predicts future 3-month revenue trends using time series analysis to support inventory planning, staffing decisions, and financial projections.")

forecast_file = os.path.join(ML_RESULTS_DIR, 'sales_forecast.csv')
if os.path.exists(forecast_file):
    try:
        forecast_data = pd.read_csv(forecast_file)
        
        historical = forecast_data[forecast_data['Type'] == 'Historical']
        future = forecast_data[forecast_data['Type'] == 'Forecast']
        
        # Metrics
        last_month_actual = historical.iloc[-1]['Revenue']
        first_forecast = future.iloc[0]['Predicted_Revenue']
        forecast_growth = ((first_forecast - last_month_actual) / last_month_actual) * 100
        total_forecast = future['Predicted_Revenue'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Last Month Revenue", f"${last_month_actual:,.0f}")
        col2.metric("Next Month Forecast", f"${first_forecast:,.0f}", f"{forecast_growth:+.1f}%")
        col3.metric("3-Month Forecast", f"${total_forecast:,.0f}")
        col4.metric("Forecast Horizon", "3 Months")
        
        # Forecast visualization
        fig_forecast = go.Figure()
        
        # Historical actual
        fig_forecast.add_trace(go.Scatter(
            x=historical['YearMonth'],
            y=historical['Revenue'],
            mode='lines+markers',
            name='Actual Revenue',
            line=dict(color='#3182ce', width=3),
            marker=dict(size=6)
        ))
        
        # Historical predicted (model fit)
        fig_forecast.add_trace(go.Scatter(
            x=historical['YearMonth'],
            y=historical['Predicted_Revenue'],
            mode='lines',
            name='Model Prediction (Fitted)',
            line=dict(color='#805ad5', width=2, dash='dash'),
            opacity=0.7
        ))
        
        # Future forecast
        fig_forecast.add_trace(go.Scatter(
            x=future['YearMonth'],
            y=future['Predicted_Revenue'],
            mode='lines+markers',
            name='Revenue Forecast',
            line=dict(color='#f56565', width=3),
            marker=dict(size=10, symbol='diamond')
        ))
        
        fig_forecast.update_layout(
            title='Revenue Forecast: Historical vs Predicted',
            xaxis_title='Time Period',
            yaxis_title='Revenue ($)',
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_forecast = style_chart(fig_forecast)
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast table
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📅 3-Month Revenue Forecast:**")
            forecast_display = future[['YearMonth', 'Predicted_Revenue']].copy()
            forecast_display['Predicted_Revenue'] = forecast_display['Predicted_Revenue'].apply(lambda x: f"${x:,.2f}")
            forecast_display.columns = ['Month', 'Predicted Revenue']
            st.dataframe(forecast_display, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**📊 Model Performance (Historical):**")
            if 'Revenue' in historical.columns and 'Predicted_Revenue' in historical.columns:
                mape = (abs(historical['Revenue'] - historical['Predicted_Revenue']) / historical['Revenue']).mean() * 100
                rmse = ((historical['Revenue'] - historical['Predicted_Revenue']) ** 2).mean() ** 0.5
                
                perf_df = pd.DataFrame({
                    'Metric': ['Mean Absolute Percentage Error (MAPE)', 'Root Mean Squared Error (RMSE)', 'Model Accuracy'],
                    'Value': [f"{mape:.2f}%", f"${rmse:,.2f}", f"{100-mape:.2f}%"]
                })
                st.dataframe(perf_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>Forecasting Insights:</strong><br>
        • Plan inventory and resources based on predicted demand<br>
        • Adjust marketing spend to align with forecasted growth<br>
        • Monitor actual vs predicted to refine the model<br>
        • Use forecasts for financial planning and budgeting
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error loading forecast data: {str(e)}")
else:
    st.warning("Sales forecast not available. Run ML analysis to generate insights.")

st.markdown("---")

# Churn Prediction Analysis
st.subheader("⚠️ Churn Prediction: Customer Retention Analysis")
st.markdown("**Model Target:** Predicts which customers are likely to stop purchasing using XGBoost machine learning, enabling proactive retention campaigns and reducing customer churn.")

churn_file = os.path.join(ML_RESULTS_DIR, 'churn_prediction.csv')
if os.path.exists(churn_file):
    try:
        churn_data = pd.read_csv(churn_file)
        
        # Key metrics
        high_risk = len(churn_data[churn_data['ChurnRiskLevel'] == 'High'])
        medium_risk = len(churn_data[churn_data['ChurnRiskLevel'] == 'Medium'])
        low_risk = len(churn_data[churn_data['ChurnRiskLevel'] == 'Low'])
        avg_churn_risk = churn_data['ChurnRisk'].mean() * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("High Risk Customers", f"{high_risk:,}", f"{high_risk/len(churn_data)*100:.1f}%")
        col2.metric("Medium Risk Customers", f"{medium_risk:,}", f"{medium_risk/len(churn_data)*100:.1f}%")
        col3.metric("Low Risk Customers", f"{low_risk:,}", f"{low_risk/len(churn_data)*100:.1f}%")
        col4.metric("Average Churn Risk", f"{avg_churn_risk:.1f}%", "Across all customers")
        
        # Churn risk distribution
        fig_churn_dist = px.pie(
            churn_data.value_counts('ChurnRiskLevel').reset_index(),
            names='ChurnRiskLevel',
            values='count',
            title='Customer Distribution by Churn Risk Level',
            color_discrete_map={'High': '#ef5350', 'Medium': '#ffa726', 'Low': '#66bb6a'}
        )
        st.plotly_chart(fig_churn_dist, use_container_width=True)
        
        # Top at-risk customers
        st.markdown("**Top 10 Highest Churn Risk Customers:**")
        top_risk = churn_data.nlargest(10, 'ChurnRisk')[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'ChurnRisk']]
        top_risk['ChurnRisk'] = (top_risk['ChurnRisk'] * 100).round(2).astype(str) + '%'
        top_risk = top_risk.rename(columns={
            'CustomerID': 'Customer ID',
            'Recency': 'Days Since Purchase',
            'Frequency': 'Purchase Count',
            'Monetary': 'Total Spent',
            'ChurnRisk': 'Churn Risk'
        })
        st.dataframe(top_risk, use_container_width=True, hide_index=True)
        
        # Churn risk by recency
        recency_bins = [0, 30, 60, 90, 365]
        churn_data['RecencyBin'] = pd.cut(churn_data['Recency'], bins=recency_bins, labels=['0-30d', '30-60d', '60-90d', '90d+'])
        recency_churn = churn_data.groupby('RecencyBin', observed=True)['ChurnRisk'].mean() * 100
        
        fig_recency = px.bar(
            x=recency_churn.index,
            y=recency_churn.values,
            title='Average Churn Risk by Recency (Days Since Last Purchase)',
            labels={'x': 'Recency Period', 'y': 'Avg Churn Risk (%)'},
            color=recency_churn.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_recency, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>Churn Prevention Insights:</strong><br>
        • Customers with no recent purchases (90d+) have highest churn risk<br>
        • Frequency is the strongest churn indicator (engagement matters)<br>
        • Implement targeted retention campaigns for medium & high risk segments<br>
        • Use product recommendations to re-engage at-risk customers<br>
        • Monitor churn risk scores weekly to catch flight risk early
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error loading churn prediction data: {str(e)}")
else:
    st.warning("Churn prediction not available. Run ML analysis to generate insights.")

st.markdown("---")

# Product Recommendation Analysis
st.subheader("🎁 Product Recommendation: Personalized Suggestions")
st.markdown("**Model Target:** Recommends products to customers using collaborative filtering based on similar customer behavior and purchase patterns to increase cross-selling and average order value.")

rec_file = os.path.join(ML_RESULTS_DIR, 'product_recommendations.csv')
if os.path.exists(rec_file):
    try:
        rec_data = pd.read_csv(rec_file)
        
        # Key metrics
        total_recs = len(rec_data)
        unique_customers = rec_data['CustomerID'].nunique()
        unique_products = rec_data['ProductStockCode'].nunique()
        avg_score = rec_data['RecommendationScore'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Recommendations", f"{total_recs:,}")
        col2.metric("Customers Targeted", f"{unique_customers:,}")
        col3.metric("Unique Products", f"{unique_products:,}")
        col4.metric("Avg Recommendation Score", f"{avg_score:.2f}")
        
        # Top recommended products
        fig_top_products = px.bar(
            rec_data['ProductDescription'].value_counts().head(10).reset_index(),
            x='count',
            y='ProductDescription',
            orientation='h',
            title='Top 10 Most Recommended Products',
            labels={'count': 'Recommendation Count', 'ProductDescription': 'Product'},
            color='count',
            color_continuous_scale='Blues'
        )
        fig_top_products.update_layout(height=400)
        st.plotly_chart(fig_top_products, use_container_width=True)
        
        # Product affinity analysis
        affinity_file = os.path.join(ML_RESULTS_DIR, 'product_affinity.csv')
        if os.path.exists(affinity_file):
            affinity_data = pd.read_csv(affinity_file)
            
            st.markdown("**Top Product Bundle Opportunities:**")
            bundle_display = affinity_data.head(10)[[
                'Product1Description', 'Product2Description', 'CooccurrenceCount', 'BundleScore'
            ]].copy()
            bundle_display.columns = ['Product 1', 'Product 2', 'Co-Purchases', 'Bundle Opportunity']
            bundle_display['Co-Purchases'] = bundle_display['Co-Purchases'].astype(int)
            bundle_display['Bundle Opportunity'] = (bundle_display['Bundle Opportunity'] * 100).round(2).astype(str) + '%'
            st.dataframe(bundle_display, use_container_width=True, hide_index=True)
        
        # Recommendation rank distribution
        rank_dist = rec_data['Rank'].value_counts().sort_index()
        fig_rank = px.bar(
            x=rank_dist.index,
            y=rank_dist.values,
            title='Recommendation Distribution by Rank (1=Best Match)',
            labels={'x': 'Recommendation Rank', 'y': 'Count'},
            color=rank_dist.values,
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig_rank, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>Recommendation Strategy:</strong><br>
        • Use top-ranked recommendations in email campaigns (highest confidence)<br>
        • Implement 'Frequently bought together' on product pages<br>
        • Create intelligent bundles from affinity pairs for upselling<br>
        • Test recommendation impact on average order value (AOV)<br>
        • Personalize homepage recommendations using customer similarity<br>
        • Track which recommendations convert to optimize algorithm
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error loading product recommendations: {str(e)}")
else:
    st.warning("Product recommendations not available. Run ML analysis to generate insights.")

st.markdown("---")

# Action Items
st.header("🎯 Recommended Actions")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Priority Actions:</strong><br><br>
    1. <strong>Engage At-Risk Customers</strong><br>
       Deploy retention campaigns for at-risk segment<br><br>
    2. <strong>Optimize Product Placement</strong><br>
       Use market basket insights for store layout<br><br>
    3. <strong>Plan for Forecast</strong><br>
       Adjust inventory based on 3-month predictions
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="insight-box">
    <strong>💡 Growth Opportunities:</strong><br><br>
    1. <strong>Upsell Champions</strong><br>
       Introduce premium products to top customers<br><br>
    2. <strong>Cross-Sell Bundles</strong><br>
       Create bundles from association rules<br><br>
    3. <strong>Win-Back Lost Customers</strong><br>
       Target lost segment with special offers
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #718096; padding: 20px;'>
    <strong>E-Commerce Analytics Platform</strong> | Powered by Machine Learning & Advanced Analytics<br>
    Last Updated: Real-time | Data Source: Direct CSV Analysis
</div>
""", unsafe_allow_html=True)
