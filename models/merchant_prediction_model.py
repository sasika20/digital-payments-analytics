from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def train_merchant_model(df):

    X = df[["Year"]]
    y = df["Value"]

    model = RandomForestRegressor(n_estimators=200)

    model.fit(X,y)
    
    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    print("Merchant Model Performance")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    return model