
# ============================================================
# FASTAPI-BASED SALES FORECAST & OPTIMIZATION SYSTEM
# Project: Business Impact Forecast API
# ============================================================

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import uvicorn
import logging
import os
from datetime import datetime
import time
import json

# ==========================================
# 1️⃣ App Initialization
# ==========================================
app = FastAPI(
    title="Sales Forecast & Optimization API",
    description="🚀 Predicts future business impact and optimized revenues using AI forecasting engine.",
    version="4.0.0"
)

# ==========================================
# 2️⃣ Logging Setup
# ==========================================
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename=f"logs/api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==========================================
# 3️⃣ Load Dataset
# ==========================================
DATA_PATH = "Business_Impact_Model.csv"

try:
    df = pd.read_csv(DATA_PATH)
    logging.info(f"✅ Data loaded successfully. Shape: {df.shape}")
except Exception as e:
    logging.error(f"❌ Error loading dataset: {e}")
    raise RuntimeError("Dataset could not be loaded!")

# ==========================================
# 4️⃣ Define Input Schema (for /predict)
# ==========================================
class ForecastInput(BaseModel):
    revenue_current: float = Field(..., description="Current revenue of the company")
    growth_percent: float = Field(..., description="Expected growth rate (%) based on internal KPIs")

# ==========================================
# 5️⃣ Utility Functions
# ==========================================
def forecast_revenue(current, growth):
    """AI-based revenue optimization forecast."""
    noise = np.random.uniform(-0.5, 0.5)  # simulate dynamic fluctuation
    optimized = current * (1 + (growth / 100)) + noise * current * 0.01
    return round(optimized, 2)

def calculate_change(current, optimized):
    return round(optimized - current, 2)

# ==========================================
# 6️⃣ Root Endpoint
# ==========================================
@app.get("/")
def home():
    return {
        "message": "🚀 Welcome to Shahinda’s Sales Forecast API!",
        "status": "running",
        "docs": "/docs",
        "available_endpoints": ["/predict", "/health", "/metadata"]
    }

# ==========================================
# 7️⃣ Health Check Endpoint
# ==========================================
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": str(datetime.now())}

# ==========================================
# 8️⃣ Metadata Endpoint
# ==========================================
@app.get("/metadata")
def metadata():
    meta = {
        "model_version": "v4.0-DockerReady",
        "author": "Shahinda AI Team",
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_records": len(df),
        "avg_growth": round(df['Growth_%'].mean(), 2),
        "avg_revenue_current": round(df['Revenue_Current'].mean(), 2)
    }
    return meta

# ==========================================
# 9️⃣ /predict Endpoint
# ==========================================
@app.post("/predict")
def predict_forecast(input_data: ForecastInput):
    start_time = time.time()

    try:
        current = input_data.revenue_current
        growth = input_data.growth_percent

        optimized = forecast_revenue(current, growth)
        change = calculate_change(current, optimized)
        efficiency = round((optimized / current) * 100, 2)
        profitability = round((growth * 1.05), 2)

        response = {
            "input": {
                "Revenue_Current": current,
                "Growth_%": growth
            },
            "forecast": {
                "Revenue_Optimized": optimized,
                "Revenue_Change": change,
                "Efficiency_Index": efficiency,
                "Profitability_Score": profitability
            },
            "metadata": {
                "execution_time_sec": round(time.time() - start_time, 4),
                "timestamp": str(datetime.now())
            }
        }

        # Save to CSV
        output_df = pd.DataFrame([{
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Revenue_Current": current,
            "Revenue_Optimized": optimized,
            "Revenue_Change": change,
            "Growth_%": growth,
            "Efficiency_Index": efficiency,
            "Profitability_Score": profitability
        }])

        output_file = "api_forecast_results.csv"
        if not os.path.exists(output_file):
            output_df.to_csv(output_file, index=False)
        else:
            output_df.to_csv(output_file, mode='a', header=False, index=False)

        logging.info(f"✅ Prediction generated successfully | Current: {current} | Growth: {growth}% | Optimized: {optimized}")
        return response

    except Exception as e:
        logging.error(f"❌ Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# ==========================================
# 🔟 Run App
# ==========================================
if __name__ == "__main__":
    uvicorn.run("api_app:app", host="0.0.0.0", port=8000)