def clean_numeric_nulls(df):

    numeric_cols = df.select_dtypes(include=["number"]).columns

    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df