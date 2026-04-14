# 🚖 FareCast: Ride Fare Surge Analytics

# 🚀 Business Use Case : Predictive Ride Fare Surge Analytics & Market Intelligence

The primary goal of this project is to provide urban commuters, fleet managers, and mobility analysts with a predictive engine to demystify fare volatility across ride-hailing platforms like Ola, Rapido, Uber and Lyft. This project focuses on identifying and analyzing dynamic surge pricing triggers using data-driven insights. By correlating historical ride fares with spatial, temporal, and environmental factors (such as rain or peak hours), this analysis pinpoints the conditions that drive price friction. The goal is to provide actionable intelligence to optimize travel budgets, improve fleet allocation, and enhance user transparency in the mobility market.

**Wider Industry Applications:** The logic used in this project can be adapted for several high-value dynamic pricing and urban mobility environments:
* **Food Delivery & Quick Commerce** (e.g., Zomato, Swiggy, Zepto): Forecasting dynamic delivery fees, restaurant surge pricing, and peak delivery windows based on active rider supply, localized order volume, and adverse weather conditions.
* **Airline Revenue Management:** Predicting ticket price fluctuations based on booking windows, seasonality, and route demand.
* **Smart Grid Energy Pricing:** Forecasting peak energy rates based on weather forecasts, time of day, and historical consumption patterns.
* **E-Commerce & Retail:** Analyzing competitor pricing algorithms and inventory levels to dynamically adjust product pricing and maximize margins.
* **Hotel Yield Management:** Predicting room rate surges during local events, conferences, or seasonal peaks to optimize occupancy and revenue.
* **On-Demand Delivery Logistics:** Using ensemble models to predict delivery fee surges based on courier availability, weather conditions, and order volume in urban zones.

# 🛠️ The Approach & Methodology

The problem was approached as a **Dynamic Pricing and Fare Prediction** challenge. The workflow followed these key phases:

* **Data Acquisition & Cleaning:** Large-scale ride-hailing (`cab_rides.csv`) and meteorological (`weather.csv`) datasets were merged based on location and hour. The data was processed to handle missing weather values and remove extreme price anomalies using the Interquartile Range (IQR) method.
* **Feature Engineering:** Extracted temporal features from UNIX timestamps to categorize rides into distinct operational shifts (e.g., 'Morning', 'Late Night'). A robust Scikit-Learn pipeline was built using `OneHotEncoder` for categorical dimensions and `StandardScaler` for continuous variables.
* **Surge Driver Identification:** Through exploratory data analysis and machine learning evaluation, I identified that **Distance**, **Service Tier**, **Time of Day**, and **Weather Conditions** (specifically rain) are the highest predictors of fare volatility.
* **Model Architecture:** The system utilizes a modular approach, offering a **Market Overview** for macro pricing trends and provider market share, a **Fare Estimator** for real-time cost forecasting based on custom parameters, and a **Model Comparison** tool to evaluate the efficacy of algorithms like Random Forest, Gradient Boosting, and KNN.
* **Deployment:** The predictive engine is hosted via a high-fidelity **Streamlit** web application, featuring an intuitive, glassmorphism dark-themed interface designed for urban mobility and market intelligence analysis.

## ⚙️ Data Pipeline Methodology
The data engineering pipeline was designed to integrate and clean disjointed sources for a unified analysis:
* **Data Integration:** Merged the raw `cab_rides.csv` and `weather.csv` datasets by unifying UNIX timestamps into standard datetime formats and joining records based on a composite `location - date - hour` key.
* **Aggregation:** Grouped weather data by location and hour, aggregating parameters like temperature, pressure, clouds, and rain using mean values to provide an accurate contextual weather snapshot for each ride.
* **Data Cleansing:** Handled missing values by filling null rain records with zeroes and dropping rows with missing price targets.
* **Outlier Mitigation:** Applied the Interquartile Range (IQR) method (1.5 * IQR) to filter out extreme price anomalies, ensuring the models learn the true underlying surge patterns rather than noise.

## 💡 Feature Engineering
To optimize the machine learning models, several domain-specific features were extracted and engineered:
* **Temporal Segmentation:** Extracted the hour from timestamps and categorized rides into distinct operational shifts: 'Morning', 'Afternoon', 'Night', and 'Late Night'.
* **Weather Flags:** Engineered categorical weather indicators to explicitly flag 'Raining' versus 'Clear' conditions.
* **Automated Transformations:** Built a robust Scikit-Learn `Pipeline` and `ColumnTransformer`, applying `OneHotEncoder` to categorical dimensions (`cab_type`, `destination`, `source`, `name`, `day_time`) and `StandardScaler` to continuous numerical features (`distance`, `temp`, `clouds`, `pressure`, `rain`, `humidity`, `wind`, `surge_multiplier`).

## 📊 Key Business Insights
Through exploratory data analysis and model evaluation, the following operational insights were uncovered:
* **Algorithm Efficacy:** The Random Forest Regressor significantly outperformed Linear Regression, Gradient Boosting, and KNN. It effectively captured the non-linear complexities of ride-share pricing, achieving an R² score of ~0.94 for Uber and ~0.89 for Lyft.
* **Market Equilibrium:** Uber and Lyft maintain an almost perfectly balanced market share in the Boston dataset (~52% vs ~48%), confirming an equitable representation of both major ride-hailing platforms without heavy class imbalance.
* **Pricing Distribution:** Density distributions revealed that the bulk of rides occur at lower price points, with premium tiers (e.g., Black SUV, Lux Black) and weather-induced surge multipliers driving the high-value tail of the fare spectrum.

## 📂 Project Structure

```bash
RFSA/
│
├── Data/
│   ├── cab_rides.csv              # Raw Uber/Lyft ride data
│   ├── weather.csv                # Hourly Boston weather data
├── RFSA_Uber&Lyft-Boston.ipynb    # Jupyter Notebook with EDA & Model Training
├── app.py                         # Main Streamlit Application script
├── label_encoders.pkl             # Serialized categorical encoders 
└── surge_pricing_model.pkl        # Pre-trained Random Forest model
└── requirements.txt               # Project Dependencies
```

## 🛠️ Tech Stack
This project utilizes a modern data science stack to process, model, and visualize ride-share data:
* **Programming Language:** Python
* **Data Processing & Analytics:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest, Gradient Boosting, K-Nearest Neighbors, Linear Regression)
* **Interactive Dashboards:** Streamlit
* **Visualizations:** Plotly Express, Seaborn, Matplotlib

---

## 🏗️ How to Run

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/MTank76/FareCast.git
    cd Farecast
    ```

2. **Setup Virtual Environment**
    ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
    
3.  **Install Dependencies**
    ```bash
     pip install -r requirements.txt
    ```

4.  **Run the Notebook:**
    Open `RFSA_Uber&Lyft-Boston.ipynb` in Jupyter Notebook or VS Code and run all cells.

5. Ensure the `Data/` directory contains the `cab_rides.csv`, `weather.csv`, and background assets (like `online taxibg.avif`).

6. Ensure the serialized model files (`surge_pricing_model.pkl` and `label_encoders.pkl`) are in the root directory.

7.  **Fire up the Dashboard**
    ```bash
    streamlit run app.py
     ```

> **Quick Start:** `pip install -r requirements.txt && streamlit run app.py`

---

## 🤝 References

  * Dataset Source: [Uber & Lyft Cab Prices](https://www.kaggle.com/datasets/ravi72munde/uber-lyft-cab-prices)
  * Inspired by: [How Ride-Hailing Apps Predict Prices & Demand](https://www.youtube.com/watch?v=HZrIAXmKigE)
  * Refrence:- [Ride-hailing fares and demand interactions: insights from market analysis over space and time](https://doi.org/10.1016/j.retrec.2026.101738)
