Rossmann Sales Forecasting: EDA & Time Series Analysis

Exploratory data analysis and 6-week demand forecasting on daily sales data from 1,115 Rossmann drugstores across Europe, built in Python on Google Colab.

# About the Dataset

Historical daily sales data from the Rossmann Store Sales Kaggle competition:

train.csv — ~1M daily records (Jan 2013 – Jul 2015): Store, Date, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday
test.csv — same structure, held out for prediction (no Sales column)
store.csv — store-level metadata: StoreType, Assortment, CompetitionDistance, Promo2 details

# What This Notebook Covers
Data cleaning — merging store metadata, handling missing values, filtering closed days
Exploratory analysis — sales trends over time, day-of-week patterns, promo/holiday/store-type effects
Trend & seasonality decomposition (weekly cycle vs. long-term trend vs. noise)
Demand forecasting using Prophet, projecting total daily sales 6 weeks ahead
Model evaluation — holding out real data to measure forecast accuracy (MAE, MAPE)

# Tools Used
Python (pandas, numpy)
Matplotlib, Seaborn, Plotly (visualization)
statsmodels (seasonal decomposition)
Prophet (forecasting)
Google Colab
📁 Repository Contents
├── rossmann-eda-forecasting-colab.ipynb   # Main analysis notebook
├── train.csv                              # Historical daily sales
├── test.csv                               # Forecast/prediction period
├── store.csv                              # Store metadata
└── README.md                              # This file

# How to Run
Clone this repo
Open rossmann-eda-forecasting-colab.ipynb in Google Colab
Upload train.csv, test.csv, and store.csv when prompted
Run all cells (Runtime → Run all)

# Key Findings
Total sales: ~$5.87B across 1,115 stores over 2.5 years (Jan 2013 – Jul 2015), averaging ~$6,934/store/day when open
Promotions work: average sales jump from $5,929 to $8,228/day (+39%) on promo days
Store Type B significantly outperforms the other three types ($10,231/day vs. ~$6,800–6,900 for A, C, D), despite being the least common store type
Assortment B (extended product range) drives the highest average sales ($8,639/day), followed by Assortment C and A
Weekly pattern: Sunday (Day 7) and Monday (Day 1) are the strongest sales days (~$8,200/day); Saturday is the weakest (~$5,875/day)
School holidays correlate with slightly higher sales ($7,200 vs. $6,897 on non-holidays), likely due to families shopping while out of school
Sales and foot traffic move together: a strong correlation (0.82) between daily Sales and Customers confirms that most of the sales variation is driven by store visits rather than basket size
The forecast model's accuracy on held-out data is reported directly in the notebook via MAE/MAPE — see Section 8

# Notes & Limitations
The forecast in this notebook is built on aggregate sales across all stores, not per-store. Per-store forecasting is flagged as a future improvement, since store-level demand patterns vary meaningfully (see Store Type/Assortment findings above)
CompetitionDistance and Promo2 fields have some missing values in store.csv, handled via median/zero-fill in the cleaning step — see notebook for details
🔭 Next Steps
Forecast per store (loop Prophet per Store, or use a global model with store as a feature)
Add Promo, StateHoliday, and SchoolHoliday as Prophet regressors
Compare against SARIMA or LightGBM with lag features
Generate a submission-style forecast against the real test.csv
