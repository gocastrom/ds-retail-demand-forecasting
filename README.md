# ds-retail-demand-forecasting
Retail demand forecasting per SKU and store using time-series features and ML (lag features, rolling windows, promotions). End-to-end pipeline from raw data → features → forecasting models → evaluation.

# Retail Demand Forecasting 🛒📈

End-to-end Machine Learning pipeline to forecast product demand per SKU and store using public retail sales data.

This project simulates a real retail analytics use case:
**predict daily unit sales 7 days ahead to improve inventory planning and reduce stockouts.**

---

## 🚀 Business Problem

Retail operations require accurate short-term demand forecasts to:

- Reduce stockouts
- Optimize replenishment
- Lower inventory holding costs
- Improve promotion planning
- Increase on-shelf availability

We forecast:

> unit_sales(t + 7)  
per (store, SKU)

---

## 📦 Dataset

Public retail dataset (Kaggle – Favorita Grocery Sales).

Granularity:
- Store
- SKU (item)
- Date (daily)
- Unit sales
- Promotions
- Metadata (store & item attributes)

Scale:
- Millions of rows
- Thousands of SKUs
- Hundreds of stores

---

## 🧠 Approach

### 1. Data Engineering
- CSV → Parquet
- joins (items + stores)
- sampling for local development

### 2. Feature Engineering (time-series)
Per SKU & Store:
- lag_1, lag_7, lag_14, lag_28
- rolling mean (7, 28 days)
- day of week
- month
- promotion flag

### 3. Modeling
Baseline:
- Naive lag

ML:
- HistGradientBoostingRegressor (sklearn)

### 4. Evaluation
- MAE (Mean Absolute Error)
- Temporal split (train past → test future)

---

## 📊 Results (sampled training)

| Model | MAE |
|-------|-------|
| Baseline (lag-1) | X.XX |
| HGB Regressor | X.XX |

> Lower MAE = better demand forecast accuracy

---

## 🏗️ Project Structure

