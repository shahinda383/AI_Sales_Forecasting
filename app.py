
# ============================================================
# DOCKERIZED BUSINESS IMPACT FORECAST SYSTEM
# Project: Sales Forecasting & Optimization
# ============================================================

import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime

# ==========================================
# 1️⃣ Logging Setup
# ==========================================
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename=f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("🚀 Docker ML System started successfully")

# ==========================================
# 2️⃣ Load Dataset
# ==========================================
DATA_PATH = "Business_Impact_Model.csv"

try:
    df = pd.read_csv(DATA_PATH)
    logging.info(f"✅ Data loaded successfully. Shape: {df.shape}")
except Exception as e:
    logging.error(f"❌ Error loading data: {e}")
    raise

# ==========================================
# 3️⃣ Data Validation
# ==========================================
required_columns = ['Date', 'Revenue_Current', 'Revenue_Optimized', 'Revenue_Change', 'Growth_%']

missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    logging.error(f"❌ Missing columns in dataset: {missing_cols}")
    raise ValueError(f"Dataset missing columns: {missing_cols}")

# ==========================================
# 4️⃣ Business Intelligence Metrics
# ==========================================
df['Date'] = pd.to_datetime(df['Date'])
df['Efficiency_Index'] = (df['Revenue_Optimized'] / df['Revenue_Current']) * 100
df['Profitability_Score'] = np.where(df['Revenue_Change'] > 0, df['Growth_%'] * 1.1, df['Growth_%'] * 0.9)

summary = {
    "Start_Date": str(df['Date'].min().date()),
    "End_Date": str(df['Date'].max().date()),
    "Total_Records": len(df),
    "Avg_Growth_%": round(df['Growth_%'].mean(), 2),
    "Avg_Efficiency_Index": round(df['Efficiency_Index'].mean(), 2),
    "Avg_Profitability_Score": round(df['Profitability_Score'].mean(), 2)
}

logging.info("📊 Summary Metrics calculated successfully")
print(json.dumps(summary, indent=4))

# ==========================================
# 5️⃣ Advanced Feature: Auto Health Check
# ==========================================
def health_check():
    health_status = {
        "status": "healthy" if len(df) > 0 else "unhealthy",
        "last_update": str(datetime.now())
    }
    with open("system_health.json", "w") as f:
        json.dump(health_status, f, indent=4)
    logging.info("🩺 Health check file created")

health_check()

# ==========================================
# 6️⃣ Advanced Feature: Automated CSV Export
# ==========================================
output_path = "business_impact_results.csv"
df.to_csv(output_path, index=False)
logging.info(f"📁 Results saved to {output_path}")

# ==========================================
# 7️⃣ Metadata Info
# ==========================================
metadata = {
    "Model_Version": "v3.0-Docker",
    "Generated_On": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Author": "Data Gems Team",
    "Environment": "Docker Container",
    "Notes": "Fully reproducible forecasting system inside container"
}

with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

logging.info("🧠 Metadata saved successfully")
print("\n✅ System executed successfully inside Docker container!")
print(f"📁 Output file: {output_path}")