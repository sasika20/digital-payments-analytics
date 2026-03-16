import pandas as pd

def fill_missing_months(df):

    full_dates = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="MS"
    )

    df = df.reindex(full_dates)

    df["Value"] = df["Value"].interpolate()

    return df