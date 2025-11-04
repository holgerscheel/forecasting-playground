# 📈 Forecasting Playground

**Forecasting Playground** is an interactive Streamlit app designed for teaching and exploring **operations analytics and forecasting methods**. 

🚀 **Try it live:**  [https://forecasting-playground.streamlit.app](https://forecasting-playground.streamlit.app)

It provides a no-code interface for running and comparing classical time series forecasting techniques (single, double, triple exponential smoothing SES, DES, TES) and machine learning models (Regression Tree, Linear Regression, XGBoost) on user-supplied datasets.

---

## 🚀 Features

- **Upload & prepare data**
  - Accepts `.xlsx`, `.xls`, or `.csv` files.  
  - Automatically detects separators (`,` or `;` for CSV).  
  - Checks for numeric columns and creates a canonical `row_index`.  
  - Displays a preview and validates data consistency.

- **Choose forecasting methods**
  - Simple Exponential Smoothing (SES)  
  - Double Exponential Smoothing (DES)  
  - Triple Exponential Smoothing (TES)  
  - Regression Tree  
  - Linear Regression  
  - XGBoost Regressor  

- **Interactive parameter control**
  - Tune model-specific hyperparameters via sliders and checkboxes.  
  - Control train/test split ratio (default: 80/20).  
  - Enable or disable automatic parameter estimation for exponential smoothing.

- **Model training & comparison**
  - Train models individually or all at once.  
  - View training and test performance (MSE, MAPE).  
  - Visualize forecasts over the training and test range.  
  - Optional tree structure visualization for the Regression Tree model.

- **Export results**
  - Download forecasts as Excel or CSV files.  
  - Clear stored results interactively.  
  - Exportable comparison tables for all metrics.

---

## 📊 Input Data Requirements

To ensure correct operation, your uploaded file should meet the following conditions:

- **Sorted in time order** (chronologically).  
  The app assumes the rows represent consecutive time steps (e.g., months, days, or weeks).  
  No explicit date column is required; the app automatically assigns a numeric `row_index` column.

- **At least 10 data points** are required.

- **Contains at least one numeric column** for forecasting.

- **Optional:** Additional numeric columns can serve as **features** for ML models (Regression Tree, XGBoost, Linear Regression). Non-numeric columns will be ignored. Date columns are supported and automatically converted into numeric day counts (days since first date).

Example structure:

| Month  | Demand  | Temperature | Marketing_Spend |
|--------|---------|-------------|-----------------|
| 1      | 120     | 3.5         | 2000            |
| 2      | 135     | 4.2         | 2500            |
| 3      | 150     | 6.1         | 3000            |
| ...    | ...     | ...         | ...             |



## 🧠 How to Use


1. **Upload your dataset** (Excel or CSV).  
   Choose the target variable to forecast.

2. **Select forecasting methods** to include in the experiment.

3. **Adjust parameters** as desired for each method, or skip this and keep defaults.  
   Then **Train models** — either individually or all at once.

4. **Review results:**
   - View accuracy metrics (MSE, MAPE)
   - Compare forecast plots across models
   - Export forecast results for external analysis

---

## 🖼 Example Output

- **Forecast Comparison Chart**  
  Displays actual values and model predictions with a shaded test range.

- **Metrics Table**  
  Interactive and sortable performance table with model parameters and coefficients.

- **Regression Tree Visualization**  
  If enabled, a zoomable decision tree graphic for model interpretation.

---

## 📂 Output Files

- `forecast_results.xlsx` – Excel file with actual and predicted values for each model  
- `forecast_results.csv` – Same data in CSV format  

---

## 🧑‍🏫 Teaching Context

This app was developed for use in **Operations Management & Analytics** courses.  
It emphasizes *conceptual understanding* of model behavior, tradeoffs, and parameter sensitivity — not coding efficiency.  
Students can experiment with model settings visually, without writing any code.

---

## ⚙️ Default Parameters

| Method | Key Parameters |
|---------|----------------|
| SES | α (smoothing level), auto optimization |
| DES | α, β (trend), auto optimization |
| TES | α, β, γ, seasonal_periods |
| Regression Tree | max_depth, n_lags, min_samples_split |
| Linear Regression | intercept, selected features |
| XGBoost | n_estimators, max_depth, learning_rate, n_lags |

---

## 🧩 Local installation (optional)

If you don't want to use [https://forecasting-playground.streamlit.app](https://forecasting-playground.streamlit.app) but install it locally or modify code:

### 1. Python Environment

Install the required packages:

```bash
pip install streamlit pandas numpy scikit-learn statsmodels xgboost matplotlib openpyxl
```

### 2. **Run the app** from your terminal:
   ```bash
   streamlit run app.py
   ```



## 🧾 License

MIT License — feel free to use, modify, and adapt for educational purposes.

Copyright (c) 2025 Holger Scheel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
