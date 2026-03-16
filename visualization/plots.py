import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")

def plot_upi_trend(df):

    plt.figure()

    plt.plot(df.index, df["Value"])

    plt.title("UPI Transaction Trend")

    plt.xlabel("Date")

    plt.ylabel("Transaction Value (Cr)")

    plt.show()

def plot_top_apps(df, year):

    year_df = df[df["Year"] == year]

    top_apps = (
        year_df
        .sort_values("Predicted_Volume", ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10,6))

    plt.bar(top_apps["App"], top_apps["Predicted_Volume"])

    plt.title(f"Top 10 Predicted UPI Apps ({year})")
    plt.xlabel("UPI Apps")
    plt.ylabel("Predicted Transaction Volume")

    plt.xticks(rotation=35)

    plt.tight_layout()
    plt.show()


def plot_trend(df, app):

    app_df = df[df["App"] == app]

    plt.figure(figsize=(8,5))

    plt.plot(app_df["Year"], app_df["Predicted_Volume"], marker="o")

    plt.title(f"{app} Transaction Prediction")
    plt.xlabel("Year")
    plt.ylabel("Predicted Volume")

    plt.grid(True)

    plt.show()

def plot_upi_forecast(actual, forecast):

    plt.figure()

    plt.plot(actual.index, actual["Value"], label="Actual")

    plt.plot(forecast["Date"], forecast["Predicted_UPI_Value"], label="Forecast")

    plt.title("UPI Forecast")

    plt.xlabel("Date")

    plt.ylabel("Transaction Value")

    plt.legend()

    plt.show()

# def plot_app_volume(apps):

#     # use only latest year
#     latest_year = apps["Year"].max()
#     apps_latest = apps[apps["Year"] == latest_year]

#     # total volume per app
#     app_totals = apps_latest.groupby("App")["Volume"].sum()

#     # top 10 apps
#     top_apps = app_totals.sort_values(ascending=False).head(10)

#     plt.figure(figsize=(10,6))

#     bars = plt.bar(top_apps.index, top_apps.values, color="#2E86C1")

#     plt.title(f"Top 10 UPI Apps by Volume ({latest_year})", fontsize=14)
#     plt.xlabel("App")
#     plt.ylabel("Transaction Volume (Mn)")

#     plt.xticks(rotation=30, ha="right")

#     plt.grid(axis="y", linestyle="--", alpha=0.5)

#     plt.tight_layout()
#     plt.show()

def plot_merchant_distribution(merchant):

    # remove unwanted rows
    merchant = merchant[
        ~merchant["Category"].str.lower().isin(["total", "others"])
    ]

    merchant_totals = merchant.groupby("Category")["Value"].sum()

    top_categories = merchant_totals.sort_values(ascending=False).head(10)

    plt.figure(figsize=(12,7))

    top_categories.plot(kind="barh", color="darkgreen")

    plt.title("Top 10 Merchant Categories")
    plt.xlabel("Transaction Value")
    plt.ylabel("Category")

    plt.gca().invert_yaxis()   # biggest on top

    plt.tight_layout()
    plt.show()