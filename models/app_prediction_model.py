from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def train_app_model(df):

    # Encode App names
    encoder = LabelEncoder()

    df["App_ID"] = encoder.fit_transform(df["App"])

    X = df[["Year","App_ID"]]

    y = df["Volume"]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X,y)

   
    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    print("App Model Performance")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)
    return model, encoder