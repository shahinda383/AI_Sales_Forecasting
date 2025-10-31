
# ==========================================================
# STREAMLIT INTERACTIVE DASHBOARD + FORECAST SYSTEM
# Project: Business Impact & Optimization Intelligence
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import io

# ==========================================================
# 1️⃣ PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="Business Impact Forecast Dashboard",
    layout="wide",
    page_icon="📊",
)

st.title("📊 Business Impact Intelligence Dashboard")
st.markdown("### Predict. Optimize. Grow. 🚀")
st.markdown("---")

# ==========================================================
# 2️⃣ FILE UPLOAD SECTION
# ==========================================================
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

if uploaded_file:
    df = load_data(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    # ======================================================
    # 3️⃣ DATA OVERVIEW
    # ======================================================
    st.subheader("🔍 Data Overview")
    st.write(df.head())

    # KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        total_gain = df["Revenue_Change"].sum()
        st.metric("💰 Total Revenue Change", f"{total_gain:,.0f} EGP")
    with col2:
        avg_growth = df["Growth_%"].mean()
        st.metric("📈 Average Growth %", f"{avg_growth:.2f}%")
    with col3:
        last_revenue = df["Revenue_Optimized"].iloc[-1]
        st.metric("🏦 Last Optimized Revenue", f"{last_revenue:,.0f} EGP")

    st.markdown("---")

    # ======================================================
    # 4️⃣ VISUALIZATIONS
    # ======================================================
    st.subheader("📊 Visual Insights")

    fig1 = px.line(df, x="Date", y=["Revenue_Current", "Revenue_Optimized"],
                   title="Revenue Trend (Current vs Optimized)",
                   labels={"value": "Revenue (EGP)", "Date": "Date"})
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(df, x="Date", y="Revenue_Change",
                  title="Revenue Change Over Time",
                  color="Revenue_Change", color_continuous_scale="Viridis")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(df, x="Revenue_Current", y="Growth_%", trendline="ols",
                      title="Growth % vs Current Revenue")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ======================================================
    # 5️⃣ FORECASTING MODULE (Advanced)
    # ======================================================
    st.subheader("🤖 AI Forecasting Model")

    df = df.sort_values("Date")
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Revenue_Optimized"].values

    model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=3)
    model.fit(X, y)

    # Forecast next 12 points
    future_steps = 12
    future_X = np.arange(len(df), len(df) + future_steps).reshape(-1, 1)
    predictions = model.predict(future_X)

    forecast_dates = pd.date_range(df["Date"].iloc[-1], periods=future_steps + 1, freq="M")[1:]
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Forecasted_Revenue": predictions
    })

    fig4 = px.line(forecast_df, x="Date", y="Forecasted_Revenue",
                   title="🔮 Forecasted Revenue for Next 12 Months",
                   line_shape="spline")
    st.plotly_chart(fig4, use_container_width=True)

    # Model Performance
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)

    st.write(f"*Model Performance:* R² = {r2:.4f}, MAE = {mae:.2f}, RMSE = {rmse:.2f}")

    st.markdown("---")

    # ======================================================
    # 6️⃣ DOWNLOAD FORECAST RESULTS
    # ======================================================
    st.subheader("💾 Export Forecast Results")

    merged_results = pd.concat([df[["Date", "Revenue_Optimized"]], forecast_df], ignore_index=True)
    csv_buffer = io.StringIO()
    merged_results.to_csv(csv_buffer, index=False)

    st.download_button(
        label="📥 Download Forecast Report (CSV)",
        data=csv_buffer.getvalue(),
        file_name="Forecast_Report.csv",
        mime="text/csv",
    )

    st.success("✅ Forecast report ready for download!")

else:
    st.info("👆 Please upload a CSV file to begin analysis.")

st.markdown("---")
st.caption("Built with ❤ by Shahinda Team | Powered by Streamlit, XGBoost, and MLflow.")