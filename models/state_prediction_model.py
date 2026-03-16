from sklearn.ensemble import RandomForestRegressor

def train_state_model(df):

    X = df[["Year"]]
    y = df["Value"]

    model = RandomForestRegressor()

    model.fit(X,y)

    return model