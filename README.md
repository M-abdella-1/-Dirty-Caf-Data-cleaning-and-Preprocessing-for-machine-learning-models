[README.md](https://github.com/user-attachments/files/28421774/README.md)
# ☕ Café Sales Data Cleaning & Feature Engineering

> A complete pipeline for cleaning a real-world dirty café sales dataset, performing
> exploratory data analysis, engineering ML-ready features, and exporting a clean CSV
> ready for modelling.

---

## 📌 1. Problem Statement

Raw transactional data collected from café point-of-sale systems is rarely clean.
Values go missing, numeric fields get corrupted with text, placeholder strings like
`"UNKNOWN"` or `"ERROR"` pollute the data, and date parsing fails silently.

**The goal of this project** is to transform a messy, real-world café sales dataset
(`dirty_cafe_sales.csv`) into a fully clean, analysis-ready DataFrame that can be used to:

- Understand sales patterns (what sells, when, where, and how)
- Build machine learning models to predict revenue or classify high-spending transactions
- Generate monthly time-series features for revenue forecasting

---

## 🐛 2. Dataset Issues

The raw dataset contained the following data quality problems:

| # | Issue | Affected Columns | Description |
|---|-------|-----------------|-------------|
| 1 | **Placeholder strings** | All columns | Values `"UNKNOWN"` and `"ERROR"` used instead of proper nulls |
| 2 | **Numeric columns stored as text** | `Quantity`, `Price Per Unit`, `Total Spent` | String dtype prevented arithmetic operations |
| 3 | **Missing product names** | `Item` | Rows with no product name are unrecoverable |
| 4 | **Missing prices** | `Price Per Unit` | Cannot calculate totals without price |
| 5 | **Missing quantities** | `Quantity` | Derivable from `Total Spent / Price Per Unit` when possible |
| 6 | **Missing totals** | `Total Spent` | Derivable from `Quantity × Price Per Unit` when possible |
| 7 | **Missing location & payment** | `Location`, `Payment Method` | Categorical nulls needing group-based imputation |
| 8 | **Unparsed dates** | `Transaction Date` | Raw strings not converted to datetime; temporal features not extracted |
| 9 | **Potential outliers** | `Quantity`, `Price Per Unit`, `Total Spent` | Extreme values detected via IQR method |

---

## 🧹 3. Cleaning Steps

The pipeline follows a **preserve-first, drop-last** strategy: we recover as many rows
as possible through imputation before resorting to deletion.

### Step 1 — Replace Invalid Strings with NaN
```python
invalid_values = ["UNKNOWN", "ERROR"]
df.replace(invalid_values, np.nan, inplace=True)
```
Converts placeholder strings into proper `NaN` so Pandas treats them as missing values.

---

### Step 2 — Cast Numeric Columns
```python
for col in ["Quantity", "Price Per Unit", "Total Spent"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
```
Forces numeric conversion; any value that cannot be parsed becomes `NaN`.

---

### Step 3 — Drop Unrecoverable Rows (missing `Item`)
```python
df = df.dropna(subset=['Item'])
```
A row without a product name carries no business meaning and cannot be imputed.

---

### Step 4 — Impute Price Per Unit (group median)
```python
df['Price Per Unit'] = df.groupby('Item')['Price Per Unit'].transform(
    lambda x: x.fillna(x.median())
)
```
Each item's missing price is filled with **that item's median price** — far more
accurate than a global median.

---

### Step 5 — Derive Missing Quantity
```python
# Quantity = Total Spent / Price Per Unit
df.loc[condition, 'Quantity'] = df['Total Spent'] / df['Price Per Unit']
```
When `Total Spent` and `Price Per Unit` are both available, `Quantity` can be
mathematically recovered.

---

### Step 6 — Derive Missing Total Spent
```python
# Total Spent = Quantity × Price Per Unit
df.loc[condition, 'Total Spent'] = df['Quantity'] * df['Price Per Unit']
```
Mirrors Step 5 — recovers totals rather than dropping the row.

---

### Step 7 — Drop Irrecoverable Rows
```python
df = df.dropna(subset=['Quantity', 'Total Spent'])
```
Any row where both derivation attempts failed has too little information to be useful.

---

### Step 8 — Impute Location & Payment Method (group mode)
```python
df['Location'] = df.groupby('Item')['Location'].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x)
)
df['Payment Method'] = df.groupby('Location')['Payment Method'].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x)
)
```
Categorical nulls are filled using the **most frequent value within the group** —
Location based on Item, Payment Method based on Location.

---

### Step 9 — Parse Dates & Extract Temporal Features
```python
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
df['DayOfWeek'] = df['Transaction Date'].dt.dayofweek
df['Month']     = df['Transaction Date'].dt.month
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
# ... then drop the original date column
```
Decomposes a single date into multiple numeric features that a model can learn from.

---

### Step 10 — Outlier Detection (IQR method)
```python
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
```
Flagged outliers in `Quantity`, `Price Per Unit`, and `Total Spent` for review.
No rows were dropped at this stage — bounds are stored for optional Winsorisation.

---

## 📊 4. Before vs After

| Metric | Before Cleaning | After Cleaning |
|--------|----------------|----------------|
| `Quantity` nulls | High (includes UNKNOWN/ERROR) | **0** |
| `Price Per Unit` nulls | High | **0** |
| `Total Spent` nulls | High | **0** |
| `Location` nulls | Present | **0** |
| `Payment Method` nulls | Present | **0** |
| `Transaction Date` dtype | `object` (string) | Dropped; replaced with numeric features |
| Numeric column dtypes | `object` (string) | `float64` / `Int64` |
| Temporal features | None | `DayOfWeek`, `Month`, `Day`, `Year`, `IsWeekend`, `YearMonth` |
| Data integrity check | Inconsistent financials | `Quantity × Price ≈ Total Spent` (difference ≈ 0) |

### Visualisations Explained

| Chart | Type | What It Shows |
|-------|------|---------------|
| **Box Plot — Quantity / Price / Total Spent** | Box-and-whisker | Distribution shape, median, IQR, and outlier dots beyond 1.5×IQR |
| **Top Selling Items** | Horizontal bar | The 10 most-ordered products by transaction count |
| **Revenue by Item** | Horizontal bar | The 10 products generating the most total revenue (may differ from order count) |
| **Revenue by Day of Week** | Bar | Which days of the week earn the most revenue (0=Mon, 6=Sun) |
| **Location Distribution** | Bar | How many transactions occurred at each branch/location |
| **Payment Methods** | Bar | Breakdown of how customers pay (Cash, Card, Digital, etc.) |
| **Average Revenue by Location** | Bar | Mean spend per transaction per location — separates high-volume from high-value branches |
| **Quantity Distribution** | Histogram + KDE | Shape of the order-size distribution; detects skew or multi-modal patterns |

---

## ✅ 5. Final Result

After cleaning and feature engineering, the output DataFrame (`dirty_cafe.csv`) contains:

**Original columns (cleaned):**
- `Item` — Product name
- `Quantity` — Units sold (Int64, no nulls)
- `Price Per Unit` — Unit price (float64, no nulls)
- `Total Spent` — Transaction total (float64, no nulls)
- `Location` — Branch/location (no nulls)
- `Payment Method` — Payment type (no nulls)

**Engineered temporal features:**
- `DayOfWeek` — 0 (Monday) to 6 (Sunday)
- `Month` — 1–12
- `Day` — Day of month
- `Year` — 4-digit year
- `YearMonth` — Month period (e.g. `2024-03`)
- `IsWeekend` — Binary flag: 1 if Sat/Sun

**Engineered business features:**
- `High_Spending` — Binary: 1 if `Total Spent` > median (classification target)
- `Item_Popularity` — Integer: how many times that item appears in the dataset

**Monthly time-series DataFrame (`monthly_df`):**
- `Revenue` — Total monthly revenue
- `Lag_1/2/3` — Revenue from 1, 2, 3 months ago
- `Rolling_Mean_3` — 3-month moving average
- `Rolling_Std_3` — 3-month rolling volatility
- `Growth` — Month-over-month percentage change

---

## 🛠️ 6. Tools Used

| Library | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.x | Core language |
| **Pandas** | Latest | Data loading, cleaning, transformation, groupby, imputation |
| **NumPy** | Latest | Numeric operations, NaN handling |
| **Matplotlib** | Latest | Low-level chart rendering (figure, axes, labels) |
| **Seaborn** | Latest | High-level statistical charts (barplot, boxplot, histplot) |
| **Scikit-learn** | Latest | Label encoding, scaling, train/test split, regression models, metrics |

### Install all dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 📁 7. Project Structure

```
cafe-data-cleaning/
│
├── dirty_cafe_sales.csv          # Raw input dataset (dirty)
├── dirty_cafe.ipynb              # Original notebook
├── dirty_cafe_commented.ipynb    # Fully commented notebook (this project)
├── dirty_cafe.csv                # Cleaned output dataset
└── README.md                     # This file
```

### How to Run

```bash
# 1. Clone or download the project folder
# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Launch Jupyter
jupyter notebook dirty_cafe_commented.ipynb

# 4. Run all cells top-to-bottom (Kernel → Restart & Run All)
# 5. The clean CSV will be saved as dirty_cafe.csv
```

---

## 👤 Author

**Mostafa** — Mathematics & Computer Science, Menoufia University
ML Training: National Telecommunication Institute (NTI)

> Part of an applied data science portfolio covering ML, EDA, and data cleaning services.
