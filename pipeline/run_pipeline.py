import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np

from database.db_connection import get_connection
from pipeline.update_db import update_database

from preprocessing.clean_data import clean_numeric_nulls
from preprocessing.fill_missing_months import fill_missing_months
from preprocessing.feature_engineering import create_features

from models.lstm_model import train_lstm
from models.app_prediction_model import train_app_model
from models.bank_prediction_model import train_bank_model
from models.merchant_prediction_model import train_merchant_model

from visualization.export_predictions import export_predictions

from visualization.plots import (
    plot_upi_trend,
    plot_upi_forecast,
    plot_merchant_distribution
)

import pandas as pd

from utils.clean_apps import clean_app_names
from visualization.plots import plot_top_apps, plot_trend


print("Loading prediction data...")

df = pd.read_csv("data/predictions/app_predictions.csv")

print("Cleaning app names...")
df = clean_app_names(df)


print("Removing duplicates...")

df = (
    df.groupby(["Year", "App"], as_index=False)
    ["Predicted_Volume"]
    .sum()
)


print("Showing Top Apps Prediction...")

plot_top_apps(df, 2030)




# -------------------
# UPDATE DATABASE
# -------------------

print("Checking NPCI website for updates...")
update_database()

# -------------------
# LOAD DATA
# -------------------

conn = get_connection()

upi = pd.read_sql("""
SELECT Year, Month,
[Value (In Cr#)] AS Value
FROM upi_monthly_stats
""", conn)

apps = pd.read_sql("""
SELECT
    [Year] AS Year,
    [Application Name] AS App,
    [Total_Volume (In Mn#)] AS Volume
FROM upi_apps
""", conn)

banks = pd.read_sql("""
SELECT 
    [Year] AS Year,
    [UPI Remitter Banks] AS Bank,
    [Total Volume (In Mn#)] AS Volume
FROM upi_top50_banks
""", conn)

merchant = pd.read_sql("""
SELECT 
    [Year] AS Year,
    [Description] AS Category,
    [Value (in Cr#)] AS Value
FROM upi_merchant_category
""", conn)

conn.close()

# -------------------
# PREPROCESS UPI DATA
# -------------------

upi["Year"] = upi["Year"].astype(int)

upi["date"] = pd.to_datetime(
    upi["Month"] + " " + upi["Year"].astype(str),
    format="%B %Y"
)

upi = upi.sort_values(["Year","Month"])
upi.set_index("date", inplace=True)

upi = clean_numeric_nulls(upi)
upi = fill_missing_months(upi)
upi = create_features(upi)

print("UPI data processed")

plot_upi_trend(upi)

# -------------------
# CLEAN OTHER DATASETS
# -------------------

apps = clean_numeric_nulls(apps)
apps = apps.dropna(subset=["Volume"])
apps["App"] = (
    apps["App"]
    .str.lower()
    .str.strip()
    .str.replace("#","",regex=False)
)

apps["App"] = apps["App"].replace({
    "phone pe":"PhonePe",
    "phonepe":"PhonePe",
    "google pay":"Google Pay",
    "paytm payments bank app":"Paytm",
    "paytm (iocl)":"Paytm",
    "axis bank apps":"Axis Bank",
    "super.money":"SuperMoney"
})

banks = clean_numeric_nulls(banks)
banks = banks.dropna(subset=["Volume"])

merchant = clean_numeric_nulls(merchant)
merchant = merchant.dropna(subset=["Value"])

print("Datasets cleaned")

# -------------------
# TRAIN LSTM MODEL
# -------------------

print("Training LSTM model...")
lstm_model, scaler = train_lstm(upi)

# -------------------
# UPI FUTURE FORECAST
# -------------------

print("Generating UPI future predictions...")

features = ["Value"]
data = upi[features].copy()

data["Value"] = np.log1p(data["Value"])

scaled = scaler.transform(data)

window = 12
last_data = scaled[-window:]

future_preds = []
months_to_predict = 60

for i in range(months_to_predict):

    X_input = last_data.reshape(1, window, len(features))

    pred = lstm_model.predict(X_input)

    future_preds.append(pred[0][0])

    next_row = np.array([pred[0][0]])

    last_data = np.vstack([last_data[1:], next_row])


future_preds = scaler.inverse_transform(
    np.array(future_preds).reshape(-1,1)
)[:,0]

future_preds = np.expm1(future_preds)

future_dates = pd.date_range(
    start=upi.index[-1],
    periods=months_to_predict + 1,
    freq="MS"
)[1:]

upi_future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_UPI_Value": future_preds
})

export_predictions(
    upi_future_df,
    "data/predictions/upi_future_predictions.csv"
)

plot_upi_forecast(upi, upi_future_df)

# -------------------
# TRAIN OTHER MODELS
# -------------------

print("Training app model...")
app_model, app_encoder = train_app_model(apps)

print("Training bank model...")
bank_model = train_bank_model(banks)

print("Training merchant model...")
merchant_model = train_merchant_model(merchant)

# -------------------
# APP PREDICTIONS
# -------------------

future_years = [2026, 2027, 2028, 2029, 2030]
apps_list = apps["App"].unique()

rows = []

for app in apps_list:

    app_id = app_encoder.transform([app])[0]

    for year in future_years:

        rows.append({
            "Year": year,
            "App_ID": app_id,
            "App": app
        })

future_df = pd.DataFrame(rows)

X_future = future_df[["Year","App_ID"]]

preds = app_model.predict(X_future)

future_df["Predicted_Volume"] = preds

export_predictions(
    future_df[["Year","App","Predicted_Volume"]],
    "data/predictions/app_predictions.csv"
)


# -------------------
# MERCHANT PREDICTIONS
# -------------------

future_years = pd.DataFrame({
    "Year": [2026, 2027, 2028, 2029, 2030]
})

merchant_preds = merchant_model.predict(future_years)

merchant_pred_df = pd.DataFrame({
    "Year": future_years["Year"],
    "Predicted_Value": merchant_preds
})

export_predictions(
    merchant_pred_df,
    "data/predictions/merchant_predictions.csv"
)

plot_merchant_distribution(merchant)

print("All models trained successfully")
print("Prediction files saved to data/predictions/")