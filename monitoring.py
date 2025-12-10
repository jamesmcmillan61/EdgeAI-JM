## Imports required packages 
import os
import time
import json
import psutil
import paho.mqtt.client as mqtt
from influxdb import InfluxDBClient


# gets the temperature of the CPU at that current moment
def get_cpu_temp():
    out = os.popen("vcgencmd measure_temp").read().strip()
    return float(out.replace("temp=","").replace("'C",""))

# InfluxDB and MQTT peramiters
broker = "localhost"
topic = "pi/system"
mqtt_client = mqtt.Client()
mqtt_client.connect(broker, 1883, 60)

# connects to influx
db = InfluxDBClient(host="localhost", port=8086)
db.switch_database("cpu_data")

INTERVAL_S = 10
print("System monitor running...")

while True:
    cpu_temp = get_cpu_temp()
    mem_usage = psutil.virtual_memory().percent
    cpu_load = psutil.cpu_percent()  
    
    
    payload = {"cpu_temp": cpu_temp, "mem_usage": mem_usage, "cpu_load": cpu_load}

    mqtt_client.publish(topic, json.dumps(payload))
    print("MQTT:", payload)

    point = [{
        "measurement": "system_monitor",
        "tags": {"host": "raspberrypi"},
        "fields": payload
    }]
    db.write_points(point)
    print("InfluxDB:", payload)

    time.sleep(INTERVAL_S)

