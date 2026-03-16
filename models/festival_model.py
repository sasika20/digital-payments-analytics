# from sklearn.ensemble import GradientBoostingRegressor

# def train_festival_model(df):

#     df["Lag1"] = df["Value"].shift(1)

#     df["Rolling7"] = df["Value"].rolling(7).mean()

#     df = df.fillna(0)

#     X = df[["Festival","Month_num","Lag1","Rolling7"]]

#     y = df["Value"]

#     model = GradientBoostingRegressor()

#     model.fit(X,y)

#     return model