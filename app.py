import streamlit as st
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from database_setup import log_prediction, init_db

# Configure Page
st.set_page_config(page_title="Housing Price Predictor", page_icon="🏠", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        color: #000;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.4);
    }
    .metric-card {
        background-color: #1a1c23;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database
if not os.path.exists('housing_data.db'):
    init_db()

# Load Data & Model
@st.cache_data
def load_data():
    df = pd.read_csv('housing.csv').dropna()
    return df

@st.cache_resource
def load_model():
    try:
        model = joblib.load('rf_model.pkl')
        # Patch for scikit-learn version mismatch (1.3 -> 1.4/1.5)
        if hasattr(model, 'estimators_'):
            for tree in model.estimators_:
                if not hasattr(tree, 'monotonic_cst'):
                    tree.monotonic_cst = None
        return model
    except:
        return None
df = load_data()
model = load_model()

# Header
st.title("🏠 California Housing Price Predictor")
st.markdown("Predict median house values using machine learning, backed by historical census data.")

# Sidebar Navigation
page = st.sidebar.radio("Navigate", ["📊 Dashboard & EDA", "🔮 Predict Price"])

if page == "📊 Dashboard & EDA":
    st.header("Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown('<div class="metric-card"><h3>Total Records</h3><h2>{}</h2></div>'.format(len(df)), unsafe_allow_html=True)
    col2.markdown('<div class="metric-card"><h3>Avg Median Value</h3><h2>${:,.0f}</h2></div>'.format(df['median_house_value'].mean()), unsafe_allow_html=True)
    col3.markdown('<div class="metric-card"><h3>Avg Income</h3><h2>${:,.0f}</h2></div>'.format(df['median_income'].mean() * 10000), unsafe_allow_html=True)
    col4.markdown('<div class="metric-card"><h3>Avg Rooms</h3><h2>{:,.0f}</h2></div>'.format(df['total_rooms'].mean()), unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Geographical Distribution of Prices")
    
    # Map Visualization
    fig_map = px.scatter_mapbox(
        df.sample(2000), 
        lat="latitude", lon="longitude", 
        color="median_house_value", size="population",
        color_continuous_scale=px.colors.cyclical.IceFire, 
        size_max=15, zoom=4,
        mapbox_style="carto-darkmatter"
    )
    st.plotly_chart(fig_map, use_container_width=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Income vs House Value")
        fig_scatter = px.scatter(df.sample(1000), x="median_income", y="median_house_value", 
                                 color="ocean_proximity", template="plotly_dark", opacity=0.7)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_chart2:
        st.subheader("Distribution of House Ages")
        fig_hist = px.histogram(df, x="housing_median_age", nbins=30, template="plotly_dark",
                                color_discrete_sequence=['#4facfe'])
        st.plotly_chart(fig_hist, use_container_width=True)

elif page == "🔮 Predict Price":
    st.header("Predict Housing Price")
    st.markdown("Enter the characteristics of the neighborhood below to estimate the median house value.")
    
    if model is None:
        st.error("Model is not trained yet! Please run `train_model.py`.")
    else:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                longitude = st.number_input("Longitude", value=-122.23)
                latitude = st.number_input("Latitude", value=37.88)
                age = st.number_input("Housing Median Age", value=41.0)
                rooms = st.number_input("Total Rooms", value=880.0)
                
            with col2:
                bedrooms = st.number_input("Total Bedrooms", value=129.0)
                population = st.number_input("Population", value=322.0)
                households = st.number_input("Households", value=126.0)
                income = st.number_input("Median Income (in $10k)", value=8.3252)
                
            submit = st.form_submit_button("Predict Median House Value")
            
        if submit:
            features = [longitude, latitude, age, rooms, bedrooms, population, households, income]
            prediction = model.predict([features])[0]
            
            # Log to SQLite Database
            log_prediction(features, prediction)
            
            st.success("Prediction Successful! Data logged to SQLite.")
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); 
                        padding: 30px; border-radius: 15px; border: 1px solid #4facfe; text-align: center;">
                <h3 style="color: #4facfe; margin:0;">Estimated Median Value</h3>
                <h1 style="color: #fff; margin:10px 0; font-size: 3.5rem;">${prediction:,.2f}</h1>
            </div>
            """, unsafe_allow_html=True)
