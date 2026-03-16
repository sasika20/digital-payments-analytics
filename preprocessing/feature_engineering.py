def create_features(df):

    # Month number feature
    df["Month_num"] = df.index.month

    # Festival indicator
    festival_months = [1, 10, 11, 12]

    df["Festival"] = df["Month_num"].apply(
        lambda x: 1 if x in festival_months else 0
    )

    return df