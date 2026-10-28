# 2026A8000531008_DEONISIO-YOHANES-WILFRIDUS-HORNAY_DS_A5_Time-Series-Rolling-Evaluation
DS_A5_Time Series Rolling Evaluation


# Data Science A5: Time Series Rolling Evaluation

## 1. Project Overview

This project is about time series forecasting and rolling evaluation. The main goal is to understand how a forecasting model performs over time and how its errors change when the data pattern changes.

The project uses a synthetic daily demand dataset for demonstration and analysis.

## 2. Project Structure

The project files are organized into the following folders:

- `data/` - Contains the dataset and data provenance information.
- `figures/` - Contains graphs used to show the data, prediction errors, and changes in performance.
- `notebooks/` - Contains the Jupyter Notebook used for the analysis.
- `report/` - Contains the project report.
- `results/` - Stores the results of the analysis.

Other important files include:

- `README.md` - Project introduction and instructions.
- `DATA_DICTIONARY.md` - Information about the dataset columns.
- `AI_USE_LOG.md` - Records of AI tool usage.
  
## 3. Dataset

The dataset is stored in `data/synthetic_daily_demand_demo.csv`.

It contains synthetic daily demand data for demonstration purposes. It is used to study time series patterns and evaluate forecasting performance over time.

The data provenance information is provided in `data/DATA_PROVENANCE.md`.

## 4. Method

The project uses rolling evaluation to examine forecasting performance at different points in time.

The main steps are:

1. Load and explore the dataset.
2. Check the time series patterns.
3. Make forecasts using the selected method.
4. Calculate the prediction errors.
5. Compare the errors over time and across different data patterns.
6. Visualize and discuss the results.

## 5. Figures and Results

The `figures/` folder contains visualizations of the time series, MAE by origin, prediction errors over time, and forecasting performance around data shifts.

The `results/` folder is used to store the analysis outputs.

These results help explain how forecasting performance changes over time.

## 6. How to Run

1. Open the Jupyter Notebook in the `notebooks/` folder.
2. Make sure Python and the required libraries are installed.
3. Run the notebook cells in order.
4. Review the generated figures and results.
5. Read the report in the `report/` folder for the main findings.

## 7. Conclusion

This project demonstrates how rolling evaluation can be used to study time series forecasting performance. It also shows why checking prediction errors over time is useful, especially when the data pattern changes.

