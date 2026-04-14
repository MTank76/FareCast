import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==========================================
# PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="FareCast | RFSA",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism Dark Mode CSS
st.markdown("""
    <style>
    /* Dark Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0B0C10 0%, #1F2833 100%);
        color: #C5C6C7;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(11, 12, 16, 0.85) !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Glassmorphism Metric Cards & Info Boxes */
    div[data-testid="metric-container"], .stAlert {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease !important;
        color: #e2e8f0 !important;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5) !important;
        background: rgba(255, 255, 255, 0.06) !important;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #66FCF1 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Chart Explanations */
    .chart-explanation {
        font-size: 0.85rem;
        color: #A0AAB2;
        background: rgba(0,0,0,0.15);
        padding: 10px 15px;
        border-radius: 8px;
        border-left: 3px solid #66FCF1;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Custom Global Color Palette (Ocean Blue & Magenta)
COLORS = {'Uber': '#00E5FF', 'Lyft': '#FF00BF'}
PLOTLY_THEME = "plotly_dark"

# ==========================================
# DATA LOADING / MOCK FALLBACK
# ==========================================
@st.cache_data(show_spinner="Loading Data...")
def load_data():
    try:
         # The direct download URL using your specific Google Drive File ID
         csv_url = "https://drive.google.com/uc?export=download&id=1q6I5CBd4Pn0I4N1SmiDa9WdLxxZIvPIx"
        # Attempt to load real data
        df_rides = pd.read_csv(csv_url)
        df_weather = pd.read_csv('Data/weather.csv')
        
        df_rides['date_time'] = pd.to_datetime(df_rides['time_stamp']/1000, unit='s')
        df_weather['date_time'] = pd.to_datetime(df_weather['time_stamp'], unit='s')
        
        df_rides['loc_date_hr'] = df_rides['source'].astype(str) + " - " + df_rides['date_time'].dt.date.astype(str) + " - " + df_rides['date_time'].dt.hour.astype(str)
        df_weather['loc_date_hr'] = df_weather['location'].astype(str) + " - " + df_weather['date_time'].dt.date.astype(str) + " - " + df_weather['date_time'].dt.hour.astype(str)
        
        weather = df_weather.groupby(['loc_date_hr']).agg({
            'temp': 'mean', 'clouds': 'mean', 'pressure': 'mean', 
            'rain': 'mean', 'humidity': 'mean', 'wind': 'mean'
        }).reset_index()
        weather['location'] = df_weather.groupby(['loc_date_hr'])['location'].first().values
        weather['rain'] = weather['rain'].fillna(0)
        
        merged_df = df_rides.merge(weather, on='loc_date_hr', how='inner')
        merged_df.dropna(subset=['price'], inplace=True)
        
        merged_df['hour'] = merged_df['date_time'].dt.hour.astype(int)
        
        def categorize_time(hour):
            if 6 <= hour <= 11: return 'Morning'
            elif 12 <= hour <= 17: return 'Afternoon'
            elif 18 <= hour <= 22: return 'Night'
            else: return 'Late Night'
                
        merged_df['day_time'] = merged_df['hour'].apply(categorize_time)
        
        cols_to_drop = ['id', 'product_id', 'time_stamp', 'location', 'date_time', 'loc_date_hr', 'hour']
        merged_df.drop(columns=[col for col in cols_to_drop if col in merged_df.columns], inplace=True)
        
        # Outlier removal
        Q1 = merged_df['price'].quantile(0.25)
        Q3 = merged_df['price'].quantile(0.75)
        IQR = Q3 - Q1
        merged_df = merged_df[(merged_df['price'] >= (Q1 - 1.5 * IQR)) & (merged_df['price'] <= (Q3 + 1.5 * IQR))]
        return merged_df

    except FileNotFoundError:
        # --- ROBUST MOCK DATA GENERATOR (If CSVs are missing) ---
        np.random.seed(42)
        n = 3000
        cab_types = np.random.choice(['Uber', 'Lyft'], n)
        day_times = np.random.choice(['Morning', 'Afternoon', 'Night', 'Late Night'], n)
        distances = np.random.uniform(0.5, 7.0, n)
        names, prices = [], []
        
        for i in range(n):
            if cab_types[i] == 'Uber':
                tier = np.random.choice(['UberX', 'UberXL', 'Black', 'Black SUV', 'UberPool'])
                base = 15 if tier == 'Black SUV' else (10 if tier == 'Black' else 5)
            else:
                tier = np.random.choice(['Lyft', 'Lyft XL', 'Lux', 'Lux Black', 'Shared'])
                base = 12 if tier == 'Lux Black' else (9 if tier == 'Lux' else 5)
            names.append(tier)
            prices.append(base + (distances[i] * np.random.uniform(1.8, 3.5)))
            
        return pd.DataFrame({
            'cab_type': cab_types, 'name': names, 'price': prices, 'distance': distances,
            'day_time': day_times, 'surge_multiplier': np.random.choice([1.0, 1.25, 1.5], n, p=[0.8, 0.15, 0.05]),
            'temp': np.random.uniform(30, 60, n), 'clouds': np.random.uniform(0, 1, n),
            'pressure': np.random.uniform(990, 1020, n), 'rain': np.random.choice([0.0, 0.1, 0.5], n, p=[0.7, 0.2, 0.1]),
            'humidity': np.random.uniform(0.5, 1.0, n), 'wind': np.random.uniform(1, 15, n),
            'source': np.random.choice(['North End', 'West End', 'Fenway', 'Back Bay'], n),
            'destination': np.random.choice(['North End', 'West End', 'Fenway', 'Back Bay'], n)
        })

# ==========================================
# ML MODEL CACHING
# ==========================================
@st.cache_resource(show_spinner="Training Machine Learning Model...")
def train_model(df):
    X = df.drop('price', axis=1)
    y = df['price']
    str_attrs = ["cab_type", "destination", "source", "name", "day_time"]
    num_attrs = ["distance", "temp", "clouds", "pressure", "rain", "humidity", "wind", "surge_multiplier"]
    
    pipeline = ColumnTransformer([
        ("str", OneHotEncoder(drop="first", handle_unknown="ignore"), str_attrs),
        ("num", StandardScaler(), num_attrs)
    ])
    model = Pipeline(steps=[
        ('preprocessor', pipeline),
        ('regressor', RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1))
    ])
    model.fit(X, y)
    return model

# ==========================================
# MAIN APP NAVIGATION
# ==========================================
df = load_data()
model = train_model(df)

st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio("Select View:", ["📊 Market Overview", "🔮 Fare Estimator", "📈 Model Comparison"])
st.sidebar.markdown("---")

if page == "📊 Market Overview":
    st.title("📊 Ride Fare Surge Analytics")
    st.markdown("Dive into dynamic visualizations comparing Uber and Lyft pricing logic in Boston.")
    
    # ---------------- GLOBAL FILTERS ----------------
    st.sidebar.markdown("### 🎛️ Segment Controls")
    selected_cab = st.sidebar.multiselect("Fleet Type", df['cab_type'].unique(), default=df['cab_type'].unique())
    selected_time = st.sidebar.multiselect("Shift Period", df['day_time'].unique(), default=df['day_time'].unique())
    dist_range = st.sidebar.slider("Distance Range (Miles)", float(df['distance'].min()), float(df['distance'].max()), (0.0, 5.0))
    
    # Apply Filters
    filtered_df = df[
        (df['cab_type'].isin(selected_cab)) & 
        (df['day_time'].isin(selected_time)) &
        (df['distance'] >= dist_range[0]) & (df['distance'] <= dist_range[1])
    ]
    
    # ---------------- KPI ROW ----------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rides Analyzed", f"{len(filtered_df):,}")
    c2.metric("Average Ride Price", f"${filtered_df['price'].mean():.2f}")
    c3.metric("Max Price (Cleaned)", f"${filtered_df['price'].max():.2f}")
    c4.metric("Average Distance", f"{filtered_df['distance'].mean():.2f} mi")
    
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    
    # ---------------- CHARTS ROW 1 ----------------
    row1_c1, row1_c2 = st.columns([1, 1.5])
    
    with row1_c1:
        st.markdown("#### 🍩 Market Share")
        cab_counts = filtered_df['cab_type'].value_counts().reset_index()
        cab_counts.columns = ['Cab Type', 'Count']
        fig_pie = px.pie(cab_counts, names='Cab Type', values='Count', color='Cab Type',
                         color_discrete_map=COLORS, hole=0.5, template=PLOTLY_THEME)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+value+percent')
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)
        # Explanation below Chart
        st.markdown("<div class='chart-explanation'>💡 <b>What this means:</b> Displays the proportion of total rides fulfilled by Uber vs. Lyft based on your current filters.</div>", unsafe_allow_html=True)

    with row1_c2:
        st.markdown("#### 🎻 Price Distribution")
        fig_violin = px.violin(filtered_df, x="cab_type", y="price", color="cab_type", 
                               box=True, points=False, color_discrete_map=COLORS, template=PLOTLY_THEME)
        fig_violin.update_layout(margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="", yaxis_title="Price ($)")
        st.plotly_chart(fig_violin, use_container_width=True)
        # Explanation below Chart
        st.markdown("<div class='chart-explanation'>💡 <b>What this means:</b> Illustrates the spread of fares. Wider sections mean a higher volume of rides happened at that price. Inner boxes show median/quartile ranges.</div>", unsafe_allow_html=True)
        
    # ---------------- CHARTS ROW 2 ----------------
    row2_c1, row2_c2 = st.columns(2)
    
    with row2_c1:
        st.markdown("#### 📊 Average Price by Service Tier")
        avg_price = filtered_df.groupby(['cab_type', 'name'])['price'].mean().reset_index().sort_values(by='price')
        fig_bar = px.bar(avg_price, x='price', y='name', color='cab_type', orientation='h',
                         color_discrete_map=COLORS, template=PLOTLY_THEME)
        fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Price ($)", yaxis_title="")
        st.plotly_chart(fig_bar, use_container_width=True)
        # Explanation below Chart
        st.markdown("<div class='chart-explanation'>💡 <b>What this means:</b> Compares the mean cost across different vehicle classifications, highlighting the premium charged for luxury or larger vehicles.</div>", unsafe_allow_html=True)

    with row2_c2:
        st.markdown("#### 📏 Impact of Distance on Price")
        scatter_df = filtered_df.sample(min(len(filtered_df), 3000))
        fig_scatter = px.scatter(scatter_df, x="distance", y="price", color="cab_type", 
                                 opacity=0.6, color_discrete_map=COLORS, template=PLOTLY_THEME)
        fig_scatter.update_layout(margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Distance (Miles)", yaxis_title="Price ($)")
        st.plotly_chart(fig_scatter, use_container_width=True)
        # Explanation below Chart
        st.markdown("<div class='chart-explanation'>💡 <b>What this means:</b> Shows the correlation between distance and fare. Vertical clusters at the same distance indicate variations caused by surge pricing and service tiers.</div>", unsafe_allow_html=True)

    # ---------------- WEATHER CHART ----------------
    st.markdown("#### 🌧️ Weather Impact on Pricing")
    filtered_df['Weather'] = filtered_df['rain'].apply(lambda x: 'Raining' if x > 0.01 else 'Clear')
    weather_impact = filtered_df.groupby(['Weather', 'cab_type'])['price'].mean().reset_index()
    
    fig_weather = px.bar(weather_impact, x="Weather", y="price", color="cab_type", barmode='group',
                         color_discrete_map=COLORS, template=PLOTLY_THEME, text_auto='.2f')
    fig_weather.update_layout(margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Average Price ($)")
    st.plotly_chart(fig_weather, use_container_width=True)
    # Explanation below Chart
    st.markdown("<div class='chart-explanation'>💡 <b>What this means:</b> Compares average costs during clear vs. rainy conditions to reveal if adverse weather drives up demand and triggers higher surge pricing.</div>", unsafe_allow_html=True)

    with st.expander("📄 View Raw Filtered Dataset"):
        st.dataframe(filtered_df.head(100), use_container_width=True)


# ==========================================
# PAGE 2: PRICE PREDICTOR
# ==========================================
elif page == "🔮 Fare Estimator":
    st.title("🔮 Predictive Fare Surge Engine")
    st.markdown("Eliminating fare ambiguity through high-fidelity machine learning models trained on Boston's urban flux.")
    
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 📍 Location & Tier")
            cab_type = st.selectbox("Cab Service", ['Uber', 'Lyft'])
            available_names = df[df['cab_type'] == cab_type]['name'].unique()
            name = st.selectbox("Tier", available_names)
            source = st.selectbox("Pickup", df['source'].unique())
            destination = st.selectbox("Dropoff", df['destination'].unique())
            day_time = st.selectbox("Time", df['day_time'].unique())

        with c2:
            st.markdown("### 📏 Metrics")
            distance = st.slider("Distance (Miles)", float(df['distance'].min()), float(df['distance'].max()), float(df['distance'].mean()))
            surge_multiplier = st.selectbox("Surge", sorted(df['surge_multiplier'].unique()))
            
        with c3:
            st.markdown("### 🌧️ Weather")
            temp = st.number_input("Temp (°F)", value=float(df['temp'].mean()))
            rain = st.number_input("Rain (in)", value=0.0, step=0.01)
            humidity = st.slider("Humidity", 0.0, 1.0, float(df['humidity'].mean()))
            clouds = st.slider("Clouds", 0.0, 1.0, float(df['clouds'].mean()))
            pressure = st.number_input("Pressure", value=float(df['pressure'].mean()))
            wind = st.number_input("Wind", value=float(df['wind'].mean()))
            
    st.markdown("<br><hr>", unsafe_allow_html=True)
    
    col_btn, col_res = st.columns([1, 2])
    with col_btn:
        predict_trigger = st.button("💰 Calculate Predicted Fare", use_container_width=True, type="primary")
        
    if predict_trigger:
        input_df = pd.DataFrame({
            'distance': [distance], 'cab_type': [cab_type], 'destination': [destination], 'source': [source],
            'surge_multiplier': [surge_multiplier], 'name': [name], 'temp': [temp], 'clouds': [clouds],
            'pressure': [pressure], 'rain': [rain], 'humidity': [humidity], 'wind': [wind], 'day_time': [day_time]
        })
        prediction = model.predict(input_df)[0]
        
        with col_res:
            st.success(f"### Estimated {cab_type} ({name}) Fare: **${prediction:.2f}**")
            st.balloons()

# ==========================================
# PAGE 3: MODEL COMPARISON
# ==========================================
elif page == "📈 Model Comparison":
    st.title("📈 Machine Learning Performance")
    st.markdown("Comparison of various regression algorithms applied to this dataset.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    models = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'KNN']
    rmse_uber = [3.379, 2.176, 2.584, 2.297]
    r2_uber   = [0.844, 0.935, 0.909, 0.928]
    
    rmse_lyft = [4.409, 3.227, 3.639, 3.411]
    r2_lyft   = [0.805, 0.896, 0.867, 0.883]

    perf_df = pd.DataFrame({
        'Algorithm': models * 2,
        'Platform': ['Uber']*4 + ['Lyft']*4,
        'RMSE': rmse_uber + rmse_lyft,
        'R2': r2_uber + r2_lyft
    })

    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### 📉 Root Mean Squared Error (RMSE)")
        st.caption("Lower is better. Measures average prediction error in dollars.")
        fig_rmse = px.bar(perf_df, x='Algorithm', y='RMSE', color='Platform', barmode='group',
                          color_discrete_map=COLORS, template=PLOTLY_THEME, text_auto='.2f')
        fig_rmse.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_rmse, use_container_width=True)

    with c2:
        st.markdown("#### 🎯 R² Score (Accuracy)")
        st.caption("Higher is better. Measures how well the model explains the price variance.")
        fig_r2 = px.bar(perf_df, x='Algorithm', y='R2', color='Platform', barmode='group',
                        color_discrete_map=COLORS, template=PLOTLY_THEME, text_auto='.3f')
        fig_r2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(range=[0.6, 1.05]))
        st.plotly_chart(fig_r2, use_container_width=True)

    st.info("**Analysis:** The **Random Forest** algorithm is the strongest performer across both platforms, achieving the highest R² score (~0.93 for Uber) and the lowest dollar-error (RMSE) rate. This is why it powers the *Price Predictor* tab.")
