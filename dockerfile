
# ============================================================
# 🐳 DOCKERFILE : Sales Forecasting & Optimization System
# ============================================================

# Step 1: Base Image
FROM python:3.10-slim

# Step 2: Set working directory
WORKDIR /app

# Step 3: Copy files
COPY . /app

# Step 4: Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Step 5: Health check
HEALTHCHECK CMD ["python", "app.py"]

# Step 6: Run the application
CMD ["python", "app.py"]