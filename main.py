import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from functools import reduce
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from matplotlib.ticker import FuncFormatter
import logging

# ==============================================
# Configuration
# ==============================================
DATA_FOLDER = r'C:\Users\ASUS\Downloads'
HPI_MASTER_PATH = os.path.join(DATA_FOLDER, 'hpi_master.csv')
START_DATE = datetime(2003, 1, 1)
END_DATE = datetime(2023, 1, 1)
OUTPUT_FOLDER = 'outputs'

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

file_info = {
    "CUUR0000SEHA": {"name": "Rent CPI", "agg": "last"},
    "WPU081": {"name": "Lumber Prices", "agg": "mean"},
    "POPTHM": {"name": "Population", "agg": "last"},
    "UMCSENT": {"name": "Consumer Sentiment", "agg": "mean"},
    "GDPC1": {"name": "Real GDP", "agg": "mean"},
    "HOUST": {"name": "Housing Starts", "agg": "sum"},
    "UNRATE": {"name": "Unemployment Rate", "agg": "mean"},
    "MORTGAGE30US": {"name": "Mortgage Rate", "agg": "mean"},
    "CSUSHPISA": {"name": "Home Price Index", "agg": "last"},
    "MEHOINUSA672N": {"name": "Median Household Income", "agg": "mean"},
    "FEDFUNDS": {"name": "Federal Funds Rate", "agg": "mean"},
    "CPIAUCSL": {"name": "CPI - All Items", "agg": "mean"},
    "PERMIT": {"name": "Building Permits", "agg": "sum"},
    "EMRATIO": {"name": "Employment to Population Ratio", "agg": "mean"},
    "PSAVERT": {"name": "Personal Saving Rate", "agg": "mean"},
    "TOTALSL": {"name": "Consumer Loans", "agg": "mean"},
    "RRVRUSQ156N": {"name": "Rental Vacancy Rate", "agg": "mean"}
}

# ==============================================
# Helper Functions
# ==============================================

def format_large_numbers(x, pos):
    return f"{x / 1000:.0f}K" if x >= 1000 else f"{x:.0f}"

def save_plot(fig, filename):
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    logging.info(f"Saved plot: {filepath}")

def get_feature_target(df):
    df['lag_CSUSHPISA'] = df['CSUSHPISA'].shift(1)
    df['rolling_avg_price'] = df['CSUSHPISA'].rolling(window=3).mean()
    feature_cols = ['MORTGAGE30US', 'HOUST', 'UNRATE', 'GDPC1', 'UMCSENT', 'covid_dummy',
                    'lag_CSUSHPISA', 'rolling_avg_price',
                    'MEHOINUSA672N', 'FEDFUNDS', 'CPIAUCSL', 'PERMIT', 'EMRATIO', 'PSAVERT', 'TOTALSL', 'RRVRUSQ156N']
    hpi_cols = [col for col in df.columns if col.startswith('HPI_')]
    feature_cols.extend(hpi_cols[:2])
    df = df.dropna(subset=feature_cols + ['CSUSHPISA'])
    return df[feature_cols], df['CSUSHPISA']

# ==============================================
# Core Functions
# ==============================================

def load_and_clean_data(folder_path, hpi_path, start_date, end_date):
    all_data = []
    for file, meta in file_info.items():
        try:
            df = pd.read_csv(os.path.join(folder_path, f"{file}.csv"))
            df.columns = ['date', 'value']
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].set_index('date')
            df = getattr(df.resample('MS'), meta['agg'])()
            df.columns = [file]
            all_data.append(df)
        except Exception as e:
            logging.warning(f"Skipping {file}: {e}")
    try:
        hpi_df = pd.read_csv(hpi_path)
        hpi_df['date'] = pd.to_datetime(hpi_df['yr'].astype(str) + '-' + hpi_df['period'].astype(str).str.zfill(2) + '-01')
        hpi_df = hpi_df[(hpi_df['date'] >= start_date) & (hpi_df['date'] <= end_date)]
        hpi_df = hpi_df[hpi_df['level'] == 'USA or Census Division']
        grouped = hpi_df.groupby(['date', 'place_name'])['index_nsa'].mean().reset_index()
        hpi_pivot = grouped.pivot(index='date', columns='place_name', values='index_nsa')
        hpi_pivot = hpi_pivot.resample('MS').mean()
        hpi_pivot.columns = [f"HPI_{col.replace(' ', '_')}" for col in hpi_pivot.columns]
        all_data.append(hpi_pivot)
    except Exception as e:
        logging.error(f"Error loading HPI master: {e}")
    if not all_data:
        raise ValueError("No data loaded")
    merged = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), all_data)
    merged = merged.interpolate(method='time').ffill(limit=3)
    logging.info("Data loading and cleaning completed.")
    return merged.reset_index()

def create_visualizations(df):
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    plot_cols = [col for col in df.columns if col in file_info or col.startswith('HPI_')]
    for i, col in enumerate(plot_cols[:9]):
        ax = axes[i // 3, i % 3]
        ax.plot(df['date'], df[col])
        ax.set_title(file_info.get(col, {}).get('name', col.replace('HPI_', '')))
        ax.grid(True)
    plt.tight_layout()
    save_plot(fig, 'time_series_subplots.png')

    plt.figure(figsize=(12, 6))
    normalized = df.set_index('date').apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    for col in ['CSUSHPISA', 'MORTGAGE30US', 'HOUST', 'UNRATE']:
        if col in normalized.columns:
            plt.plot(normalized.index, normalized[col], label=file_info.get(col, {}).get('name', col))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_plot(plt.gcf(), 'normalized_trends.png')

    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include=np.number).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    save_plot(plt.gcf(), 'correlation_matrix.png')

def model_and_analyze(df):
    df['covid_dummy'] = ((df['date'] >= '2020-03-01') & (df['date'] <= '2021-12-01')).astype(int)
    X, y = get_feature_target(df)

    tscv = TimeSeriesSplit(n_splits=5)
    ridge = Ridge(alpha=1.0)

    preds = []
    test_indices = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        ridge.fit(X_train, y_train)

        y_pred = ridge.predict(X_test)
        preds.extend(y_pred)
        test_indices.extend(test_idx)

        r2 = r2_score(y_test, y_pred)
        logging.info(f"Fold Test R2 - Ridge: {r2:.3f}")

    overall_r2 = r2_score(y.iloc[test_indices], preds)
    logging.info(f"Overall Test R2 - Ridge: {overall_r2:.3f}")

    plt.figure(figsize=(10, 6))
    plt.plot(y.iloc[test_indices].values, label='Actual', marker='o')
    plt.plot(preds, label='Predicted', marker='x')
    plt.title('Ridge Regression: Actual vs Predicted Home Price Index')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_plot(plt.gcf(), 'ridge_actual_vs_predicted.png')

    joblib.dump(ridge, os.path.join(OUTPUT_FOLDER, 'ridge_model.pkl'))
    df.to_csv(os.path.join(OUTPUT_FOLDER, 'processed_housing_data.csv'), index=False)
    logging.info("Model and data saved successfully.")

# ==============================================
# Main Execution
# ==============================================

def main():
    logging.info("Starting project pipeline...")
    data = load_and_clean_data(DATA_FOLDER, HPI_MASTER_PATH, START_DATE, END_DATE)
    create_visualizations(data)
    model_and_analyze(data)
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
