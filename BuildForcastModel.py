
# Imports -
import numpy as np
import pandas as pd
from influxdb import InfluxDBClient
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


N_LAGS = 30               
FORECAST_HORIZON = 30     
DB_NAME = "cpu_data"
MEASUREMENT = "system_monitor"


## conects to influxDB
print("Connecting to Influx db")
db = InfluxDBClient(host="localhost", port=8086)
db.switch_database(DB_NAME)

# gets data from influx DBs
query = f"""
SELECT cpu_temp, cpu_load, mem_usage
FROM {MEASUREMENT}
ORDER BY time ASC
"""
results = list(db.query(query).get_points())
df = pd.DataFrame(results)


print(df.head())
df = df.dropna()

print(df.info())


df["time"] = pd.to_datetime(df["time"])
df = df.set_index("time")




if df.empty:
    raise ValueError("No data to train on")

# Resampling 
df = df.resample("1min").mean().interpolate()

print(df.head())
print(df.info())


def make_lagged_features(df, target_col="cpu_temp", n_lags=30, horizon=30):
    X, y = [], []
    for i in range(n_lags, len(df) - horizon):
        row = []
        for col in ["cpu_temp", "cpu_load", "mem_usage"]:
            row.extend(df[col].iloc[i-n_lags:i].values)
        X.append(row)
        y.append(df[target_col].iloc[i + horizon])
    return np.array(X), np.array(y)


X, y = make_lagged_features(df, n_lags=N_LAGS, horizon=FORECAST_HORIZON)

# Splits into test and train
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Trains the model
print("Training Ridge Regression model...")
model = Ridge(alpha=0.5)
model.fit(X_train, y_train)

## predict based on test data 
y_pred = model.predict(X_test)

## evaluate the model 
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nForecasting Results:")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}\n")

# saving the model 
print("Saving model...")
joblib.dump(model, "cpu_temp_30min_forecast_model.joblib")

np.savez(
    "cpu_temp_forecast_meta.npz",
    n_lags=N_LAGS,
    horizon=FORECAST_HORIZON,
    feature_cols=["cpu_temp", "cpu_load", "mem_usage"],
    mae=mae,
    rmse=rmse,
)

print("Model generated and saved")
