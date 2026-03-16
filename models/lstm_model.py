import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def train_lstm(df):

    features = ["Value"]

    data = df[features].copy()

    data["Value"] = np.log1p(data["Value"])

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(data)

    window = 12

    X=[]
    y=[]

    for i in range(window,len(scaled)):

        X.append(scaled[i-window:i])
        y.append(scaled[i,0])

    X=np.array(X)
    y=np.array(y)

    model = Sequential()

    model.add(LSTM(
        128,
        return_sequences=True,
        input_shape=(X.shape[1],X.shape[2])
    ))

    model.add(Dropout(0.2))

    model.add(LSTM(64))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    early_stop = EarlyStopping(
        monitor="loss",
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X,
        y,
        epochs=100,
        batch_size=8,
        callbacks=[early_stop]
    )
    
    # Predict on training data
    y_pred = model.predict(X)

    # reverse scaling
    y_pred = scaler.inverse_transform(
        np.concatenate([y_pred, np.zeros((y_pred.shape[0],2))], axis=1)
    )[:,0]

    y_true = scaler.inverse_transform(
        np.concatenate([y.reshape(-1,1), np.zeros((y.shape[0],2))], axis=1)
    )[:,0]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("LSTM Model Performance")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    return model, scaler