# 📦 StockSense

### Predictive Inventory Intelligence & Decision Support

StockSense is an end-to-end machine learning project that transforms historical retail transactions into **SKU-level demand forecasts, inventory risk signals, and replenishment recommendations**.

The project combines demand forecasting with inventory decision logic to answer a practical business question:

> **Which products are likely to require replenishment, and when should action be taken?**

The final system is delivered through an interactive **Streamlit dashboard** designed to connect machine learning predictions with operational inventory decisions.

## 🎯 Business Problem

Retail businesses need to maintain enough inventory to satisfy customer demand without holding excessive stock.

Two common problems arise when inventory decisions rely mainly on historical averages or manual judgment:

- **Stockouts:** insufficient inventory can lead to missed sales and poor customer experience.
- **Overstocking:** excess inventory ties up working capital and increases inventory carrying risk.

StockSense addresses this by using historical SKU-level transaction data to:

1. Forecast near-term product demand.
2. Quantify inventory requirements using lead-time demand and safety stock.
3. Identify SKUs whose simulated inventory falls below the forecast-based reorder point.
4. Translate these signals into prioritized replenishment actions.

The goal is not simply to predict demand, but to connect **prediction → inventory risk → business action**.

## 💡 Solution

StockSense follows an end-to-end pipeline that combines machine learning with inventory management logic.

### Workflow

1. **Transaction Data**
   - Historical retail transactions are used as the primary source of demand information.

2. **Data Processing**
   - Exact duplicates, cancellations, accounting adjustments, non-product transactions, and inventory-damage records are identified and handled using explicit business rules.
   - The cleaned transactions are aggregated into daily SKU-level demand.

3. **Demand Analysis**
   - SKU demand frequency, demand magnitude, variability, and demand concentration are analyzed.
   - Intermittent demand patterns are explicitly considered.

4. **Demand Forecasting**
   - Lag, rolling-window, and calendar features are created.
   - Multiple baselines and machine learning models are evaluated using chronological validation.
   - XGBoost is used for the final demand forecasting pipeline.

5. **Inventory Decision Engine**
   - Forecast demand is combined with lead-time demand and safety stock to estimate a reorder point.
   - Simulated inventory is compared with the reorder point to identify inventory risk.

6. **Decision Support**
   - SKUs are categorized into risk levels and translated into recommended actions such as immediate reorder, reorder soon, monitoring, or no action.

7. **Streamlit Dashboard**
   - The complete pipeline is exposed through an interactive dashboard for portfolio-level and SKU-level analysis.

## 🏗️ Project Architecture

```text
                    ┌─────────────────────┐
                    │   Retail Transactions│
                    │      (UCI Dataset)   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Data Processing &  │
                    │  Business Rules     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Daily SKU Demand  │
                    │   & Demand Analysis │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │ Lags + Rolling +    │
                    │ Calendar Features   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Demand Forecasting  │
                    │      XGBoost        │
                    └──────────┬──────────┘
                               │
                     ┌─────────┴─────────┐
                     ▼                   ▼
             ┌───────────────┐   ┌────────────────┐
             │ Demand        │   │ Inventory      │
             │ Forecast      │   │ Decision Engine│
             └───────┬───────┘   └───────┬────────┘
                     │                   │
                     └─────────┬─────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Risk & Recommended  │
                    │      Actions        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Streamlit Dashboard│
                    └─────────────────────┘

## 📊 Dataset

StockSense uses the **UCI Online Retail Dataset**, containing transactions from a UK-based online retailer between December 2010 and December 2011.

### Dataset Statistics

- **541,909** raw transaction records
- **8** original columns
- **4,070** unique product codes
- Date range: **01 Dec 2010 – 09 Dec 2011**

Key fields include:

- `InvoiceNo`
- `StockCode`
- `Description`
- `Quantity`
- `InvoiceDate`
- `UnitPrice`
- `CustomerID`
- `Country`

The forecasting grain is transformed from **invoice × product line** into **SKU × day** demand.

---

## 🧹 Data Preparation & Business Rules

The raw transaction data contains records that do not represent genuine customer demand.

The preprocessing pipeline:

- Removes **5,268 exact duplicate records**.
- Identifies cancellation/return invoices beginning with `C`.
- Excludes accounting adjustments such as bad-debt records.
- Excludes non-product transaction codes such as discounts, manual adjustments, and postage.
- Excludes damaged/unsaleable inventory adjustments.
- Excludes negative-quantity, zero-price system/inventory adjustments.
- Does **not** remove all zero-price transactions because some may represent legitimate product activity.

After applying these rules, **523,888 valid demand transaction rows** remain.

The cleaned transactions are aggregated into daily SKU-level demand and expanded across each SKU's observed active period so that days with no recorded demand are represented as zero demand.

---

## 📈 Demand Analysis

The resulting demand dataset contains:

- **3,936 SKUs**
- **276,167 positive-demand SKU-days**
- **1,076,751 SKU-day records** after active-period calendar expansion

The analysis shows substantial demand intermittency and variability.

### Key Findings

- **55.03%** of SKUs have demand on 25% or fewer of their active calendar days.
- **79.75%** of SKUs have demand on 50% or fewer of their active calendar days.
- The top **20% of SKUs account for 76.88% of total unit demand**.
- Demand is strongly right-skewed, with a small number of very large transaction spikes.

Because of this demand structure, frequency and demand magnitude are treated as separate characteristics rather than relying only on average demand.

## 🔮 Demand Forecasting

The forecasting problem is formulated at the **SKU × day** level.

The initial production target is **next-day demand**, which is then used recursively to generate a **7-day demand forecast**.

### Features

The forecasting model uses 11 features:

- Lag demand: 1, 7, 14, and 28 days
- 7-day rolling mean
- 7-day rolling standard deviation
- 28-day rolling mean
- Day of week
- Month
- ISO week of year
- Weekend indicator

Rolling features are calculated using only historical observations to prevent data leakage.

### Model Selection

Several approaches were evaluated:

| Model | Validation MAE | Validation RMSE |
|---|---:|---:|
| Naive (previous day) | 11.30 | 49.12 |
| 7-Day Moving Average | 9.55 | 37.12 |
| Linear Regression | 8.80 | 35.24 |
| Tuned XGBoost | **8.52** | **34.75** |

The final forecasting model uses **XGBoost Regressor** with:

- `n_estimators = 200`
- `learning_rate = 0.03`
- `max_depth = 3`
- `subsample = 0.8`
- `colsample_bytree = 0.8`

A chronological train/validation/test strategy is used instead of random splitting because the problem is time-dependent.

### Held-Out Test Performance

The tuned XGBoost model achieved:

- **MAE: 10.25 units**
- **RMSE: 41.24 units**

On the same test period:

- Zero-demand baseline: MAE **10.45**, RMSE **46.32**
- Naive baseline: MAE **14.16**, RMSE **56.58**

The test results show that the forecasting model improves on both reference baselines while preserving the temporal structure of the problem.

---

## 📦 Inventory Decision Engine

Demand forecasts are converted into inventory decision signals using a simplified inventory planning framework.

### Assumptions

Because the UCI dataset does not contain actual inventory-on-hand or replenishment lead-time information, the project uses explicit scenario assumptions:

- **Lead time:** 7 days
- **Service level:** 95%
- **Z-score:** 1.645
- **Simulated inventory:** recent 14-day historical demand

### Reorder Point

The reorder point is calculated as:

**Reorder Point = Expected Lead-Time Demand + Safety Stock**

Where:

**Expected Lead-Time Demand = Mean Daily Demand × Lead Time**

**Safety Stock = Z × Daily Demand Standard Deviation × √Lead Time**

Simulated inventory is then compared with the forecast-based reorder point to estimate inventory gaps and prioritize SKUs requiring attention.

### Risk Classification

| Inventory Gap % | Risk Level | Recommended Action |
|---:|---|---|
| ≤ 0% | Healthy | No action |
| 0–25% | Watch | Monitor closely |
| 25–50% | High | Reorder soon |
| > 50% | Critical | Reorder immediately |

These recommendations are **decision-support outputs**, not actual purchase orders.

## ⚙️ Installation & Usage

### 1. Clone the Repository

git clone https://github.com/rajdeep-senapati/StockSense.git
cd StockSense

### 2. Create a Virtual Environment

python -m venv .venv

### 3. Activate the Virtual Environment

Windows:

.\.venv\Scripts\activate

macOS / Linux:

source .venv/bin/activate

### 4. Install Dependencies

pip install -r requirements.txt

### 5. Run the Streamlit Application

streamlit run app.py

The application will open in your browser and provide access to the StockSense dashboard.

> **Note:** The raw UCI Online Retail dataset is not included in the repository. Place the downloaded dataset at `data/raw/online_retail.xlsx` if you want to reproduce the complete pipeline from the raw data.

---

## 🔮 Future Improvements

Potential extensions include:

- Incorporating real inventory-on-hand data.
- Using SKU-specific supplier lead times.
- Adding price and promotion information.
- Testing specialized methods for intermittent demand.
- Introducing probabilistic forecasting and prediction intervals.
- Adding cost, margin, and lost-sales information to the risk engine.
- Automated model retraining as new transactions become available.
- Deployment on a cloud platform.

---

## 📚 Data Source

**UCI Online Retail Dataset**

The dataset is provided by the UCI Machine Learning Repository and is licensed under **CC BY 4.0**.
```
