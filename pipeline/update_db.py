import pandas as pd
from database.db_connection import get_connection


def update_database():

    conn = get_connection()

    print("Checking database tables...")

    # Load existing tables
    upi = pd.read_sql("SELECT Year, Month, [Value (In Cr#)] FROM upi_monthly_stats", conn)

    apps = pd.read_sql("SELECT [Year], [Application Name], [Total_Volume (In Mn#)] FROM upi_apps", conn)

    banks = pd.read_sql("SELECT [Year], [UPI Remitter Banks], [Total Volume (In Mn#)] FROM upi_top50_banks", conn)

    merchant = pd.read_sql("SELECT [Year], [Description], [Value (in Cr#)] FROM upi_merchant_category", conn)

    print("Tables loaded successfully")

    print("Rows in tables:")

    print("UPI Monthly:", len(upi))
    print("UPI Apps:", len(apps))
    print("UPI Banks:", len(banks))
    print("Merchant Category:", len(merchant))

    conn.close()