
import time
import numpy as np
import pandas as pd
import joblib
import paho.mqtt.client as mqtt
from influxdb import InfluxDBClient
from datetime import datetime, timedelta


# Get trained model
model = joblib.load("/home/u700851/edgeai_project/Coursework/cpu_temp_30min_forecast_model.joblib")
meta = np.load("/home/u700851/edgeai_project/Coursework/cpu_temp_forecast_meta.npz", allow_pickle=True)

N_LAGS = int(meta["n_lags"])
FEATURE_COLS = list(meta["feature_cols"])

# FROM
SOURCE_DB_NAME = "cpu_data"           
SOURCE_MEASUREMENT = "system_monitor" 


# TO
FORECAST_DB_NAME = "PersonDetection"  
FORECAST_MEASUREMENT = "cpu_temp_forecast"

MQTT_TOPIC = "pi/vision/forcast"        




mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(username="user1", password="test")
mqtt_client.connect("localhost", 1883, 60)

source_db = InfluxDBClient(host="localhost", port=8086)
source_db.switch_database(SOURCE_DB_NAME)

forecast_db = InfluxDBClient(host="localhost", port=8086)
forecast_db.switch_database(FORECAST_DB_NAME)


def get_latest_minutes(n):
    query = f"""
    SELECT cpu_temp, cpu_load, mem_usage
    FROM {SOURCE_MEASUREMENT}
    ORDER BY time DESC
    LIMIT {n}
    """
    points = list(source_db.query(query).get_points())
    df = pd.DataFrame(points[::-1])  
    return df

def build_features(df):
    vals = []
    for col in FEATURE_COLS:
        vals.extend(df[col].values)
    return np.array(vals).reshape(1, -1)

def write_forecast_to_influx(value):
    future_time = datetime.utcnow() + timedelta(minutes=30)
    json_body = [{
        "time": future_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "measurement": FORECAST_MEASUREMENT,
        "fields": {"forecast_30min": float(value)}
    }]
    forecast_db.write_points(json_body)


while(True):
  
  try:
    df = get_latest_minutes(N_LAGS)
    
    df = df.dropna()

    if len(df) < N_LAGS or df.empty:
      print("No data to predict on")
      sys.exit()
    else:


      X = build_features(df)



      prediction = float(model.predict(X)[0])

   
      write_forecast_to_influx(prediction)


      mqtt_client.publish(MQTT_TOPIC, str({"forecast_temp_30min": prediction}))

      print("Forecast: ")
      print(prediction)
  
  except Exception as e:
    print(f"Error occurred: {e}")
  
  time.sleep(60)
