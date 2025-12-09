
from ultralytics import YOLO
import cv2
import json
import time
import paho.mqtt.client as mqtt
from influxdb import InfluxDBClient
from picamera2 import Picamera2


# frame rate 

frame_count = 0
last_fps_time = time.time()
fps = 0
fpm = 0


# MQTT 
mqtt_broker = "localhost"
mqtt_topic = "pi/vision/human"
mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(username="user1", password="test")
mqtt_client.connect(mqtt_broker, 1883, 60)

# InfluxDB 
db = InfluxDBClient(host="localhost", port=8086)
db.switch_database("PersonDetection")

# Yolo model
model = YOLO("yolov8n.pt")

# rasbery pi camera
cam = Picamera2()
W, H = 416, 240
cfg = cam.create_video_configuration(main={"size": (W, H), "format": "YUV420"})
cam.configure(cfg)
cam.start()
time.sleep(0.5) 



# frame size
yuv = cam.capture_array()
frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
frame_height, frame_width = frame.shape[:2]


#print(f"Frame dimensions: {frame_width}x{frame_height}")

# Split into regions
left_bound = frame_width // 3
right_bound = 2 * (frame_width // 3)
#print(f"Left bound {left_bound}, Right bound {right_bound}")


while True:
    # this frame
    yuv = cam.capture_array()
    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    
    # frame rate calc         
    frame_count += 1
    current_time = time.time()
    detections_this_frame = []   

    

    elapsed = current_time - last_fps_time
    if elapsed >= 60:
      fpm = frame_count / elapsed * 60.0   
      db.write_points([{
        "measurement": "vision_stats",
        "fields": { "fpm": fpm }
      }])
    
      frame_count = 0
      last_fps_time = current_time





    # Reduce frame size and detect
    inference_frame = cv2.resize(frame, (320, int(320 * frame_height/frame_width)))
    results = model(inference_frame, stream=True, verbose=False, classes=[0],conf=0.7,imgsz=320)


    # for each person
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # select region
            if center_x < left_bound:
              region = "Left"
            elif center_x < right_bound:
              region = "Center"
            else:
              region = "Right"
            
            # Data
            result_data = {
                    "cx": center_x,
                    "cy": center_y,
                    "confidence": round(conf, 2)
            }

            follow_cmd = f"turn_{region.lower()}"


            try:
              
              mqtt_client.publish(
              mqtt_topic,
              json.dumps({**result_data, "follow_cmd": follow_cmd})
              )
            except Exception as e:
              print("Error for MQTT publsih")
            detections_this_frame.append({
                    "result_data": result_data,
                    "follow_cmd": follow_cmd
            })


    # write predictions to temp storage
    if detections_this_frame:
      
      det = detections_this_frame[0]

      point = [{
            "measurement": "ai_vision_human",
            "tags": {
                "host": "raspberrypi",
                "follow_cmd": det["follow_cmd"]
            },
            "fields": det["result_data"]
      }]

      db.write_points(point)
      
      #print("InfluxDB:", det)    
    else:
      try:
        result_data = {
                    "cx": 0,
                    "cy": 0,
                    "confidence": 0
        
        }
        # NO DETECTIONS
        mqtt_client.publish(
        mqtt_topic,
        json.dumps({**result_data, "follow_cmd": "Stop"}))

      except Exception as e:
        print("Eror with mqtt")
    

# Cleanup

cam.stop()
print("Stream ended")

