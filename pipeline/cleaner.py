def clean_data(df):

    df.columns = df.columns.str.lower().str.strip()

    df = df.drop_duplicates()

    return df