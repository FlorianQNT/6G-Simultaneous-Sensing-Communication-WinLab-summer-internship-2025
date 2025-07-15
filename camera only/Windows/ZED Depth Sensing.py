import pyzed.sl as sl
import cv2
import numpy as np
from ultralytics import YOLO
import torch


# ============================
# ADD: DeepSORT Import
# ============================
# Make sure you have installed deep-sort-realtime:
# pip install deep-sort-realtime
from deep_sort_realtime.deepsort_tracker import DeepSort


# ============================
# Initialize ZED Camera
# ============================
zed = sl.Camera()


init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA for better depth accuracy
init_params.coordinate_units = sl.UNIT.METER
init_params.sdk_verbose = 1


# Open the ZED camera
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
   print("Failed to open ZED Camera")
   exit(1)


# ============================
# Load YOLO Model (Only for People Detection)
# ============================
model = YOLO("yolov8l.pt")  # Use "yolov8x.pt" for max accuracy or "yolov8n.pt" for speed
#model = YOLO("yolo11l.pt")   # to use yolov11, need to download the version on the GitHub (not that great)
# all version are downloaded but if you want more information: https://github.com/ultralytics/ultralytics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Define class names (YOLO COCO classes)
class_names = model.names
# to enter manually the class
#person_class_id = 0  # "person" class in COCO dataset
#chair_class_id = 56  # "chair" class in COCO dataset

# slower(?) but just need to put the number of the class id
# load every class id in a list
all_class_ids = list(class_names.keys())
# table for the number of the class id needed
# to know every class: https://gist.github.com/rcland12/dc48e1963268ff98c8b2c4543e7a9be8
used_class_ids = [0]

# ============================
# Create objects and depth containers
# ============================
objects = sl.Objects()
depth_map = sl.Mat()


# ============================
# Movement Tracking Variables
# ============================
previous_depths = {}  # Store previous depth values
depth_history = {}  # Store history of depth for smoothing
depth_smoothing_window = 8  # Number of frames to smooth depth


# ============================
# Function: Depth Smoothing
# ============================
def get_stable_depth(class_id, new_depth):
   if class_id not in depth_history:
       depth_history[class_id] = [new_depth] * depth_smoothing_window  # Initialize history


   # Update depth history (keep last 'depth_smoothing_window' values)
   depth_history[class_id].append(new_depth)
   if len(depth_history[class_id]) > depth_smoothing_window:
       depth_history[class_id].pop(0)  # Remove oldest depth value


   # Compute and return the median depth value
   return np.median(depth_history[class_id])


# ============================
# ADD: Initialize DeepSORT
# ============================
# You can tweak parameters like max_age, max_iou_distance, etc.
tracker = DeepSort(
   max_age=200,        # How many frames to keep track of an object with no detections
   n_init=20,          # Number of frames required to confirm a track
   max_iou_distance=0.7
)


# ============================
# Main Processing Loop
# ============================
while zed.grab() == sl.ERROR_CODE.SUCCESS:
   # Retrieve Image Data
   image = sl.Mat()
   zed.retrieve_image(image, sl.VIEW.LEFT)
   image_np = image.get_data()


   # Retrieve Depth Data
   zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
   depth_data = depth_map.get_data()


   # Convert from RGBA to RGB if needed
   if image_np.shape[2] == 4:
       image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)


   # Run YOLO Detection
   results = model(image_np)


   # Prepare detections for DeepSORT
   deepsort_inputs = []


   if results:
       detections = results[0].boxes.xywh.cpu().numpy()  # Get xywh bounding boxes
       confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
       class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs


       for i in range(len(detections)):
           class_id = class_ids[i]


           # ✅ Filter: Only process "person" detections
           #if class_id != person_class_id:
               #continue

           # ✅ Filter: Only process "chair" detections
           #if class_id != chair_class_id:
               #continue

           # ✅ Filter: process "person" and "chair" detections
           #if class_id != chair_class_id & class_id != person_class_id :
               #continue

           # ✅ Filter: process "person" and "chair" detections with table of every class id and filter
           if class_id not in used_class_ids :
               continue

           x_center, y_center, width, height = detections[i]
           confidence = confidences[i]
           class_name = class_names[class_id]


           # Convert xywh to top-left and bottom-right
           top_left = (int(x_center - width / 2), int(y_center - height / 2))
           bottom_right = (int(x_center + width / 2), int(y_center + height / 2))


           # ============================
           # Your Existing Depth Logic
           # ============================
           center_x = int(x_center)
           center_y = int(y_center)
           raw_depth_value = depth_data[center_y, center_x]


           if np.isnan(raw_depth_value) or raw_depth_value <= 0:
               continue


           depth_value = get_stable_depth(class_id, raw_depth_value)


           if depth_value < 1.0:
               movement_threshold = 0.002
           elif depth_value < 1.5:
               movement_threshold = 0.0035
           elif depth_value < 2:
               movement_threshold = 0.0043
           elif depth_value < 3:
               movement_threshold = 0.0054
           else:
               movement_threshold = 0.07


           if class_id not in previous_depths:
               previous_depths[class_id] = depth_value


           depth_change = abs(depth_value - previous_depths[class_id])


           if depth_change > movement_threshold:
               movement = "Moving Towards" if depth_value < previous_depths[class_id] else "Moving Away"
           elif depth_change < movement_threshold / 2:
               movement = "Stationary"
           else:
               movement = "Uncertain"


           previous_depths[class_id] = depth_value


           # **Color Coding for Bounding Boxes**
           if movement == "Moving Towards":
               color = (255, 0, 0)  # Blue
           elif movement == "Moving Away":
               color = (0, 0, 255)  # Red
           else:    # Stationary or Uncertain
               color = (0, 255, 0)  # Green


           # Draw your existing bounding box
           cv2.rectangle(image_np, top_left, bottom_right, color, 3, cv2.LINE_AA)


           # Create background for text
           label = f"{class_name} ({confidence * 100:.1f}%) | {movement} | Depth: {depth_value:.2f}m"
           text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
           text_x, text_y = top_left[0], top_left[1] - 10


           cv2.rectangle(
               image_np,
               (text_x - 5, text_y - text_size[1] - 5),
               (text_x + text_size[0] + 5, text_y + 5),
               (0, 0, 0),
               -1
           )
           cv2.putText(
               image_np, label, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
               2, cv2.LINE_AA
           )


           # ============================
           # ADD (FIXED): Append to DeepSORT input
           # ============================
           # Must be ([x1, y1, x2, y2], confidence, class_id)
           x1, y1 = top_left
           x2, y2 = bottom_right
           deepsort_inputs.append(([x1, y1, x2, y2], confidence, 0))  # class_id = 0 for "person"


   # ============================
   # Update DeepSORT tracker
   # ============================
   tracks = tracker.update_tracks(deepsort_inputs, frame=image_np)


   # ============================
   # Overlay DeepSORT IDs
   # ============================
   for track in tracks:
       if not track.is_confirmed() or track.time_since_update > 150:
           continue
       track_id = track.track_id
       ltrb = track.to_ltrb()  # left, top, right, bottom
       x1, y1, x2, y2 = map(int, ltrb)


       # Draw the ID text above the bounding box
       cv2.putText(
           image_np,
           f"ID: {track_id}",
           (x1, y1 - 40),
           cv2.FONT_HERSHEY_SIMPLEX,
           0.8,
           (255, 255, 0),
           2,
           cv2.LINE_AA
       )


   # Display image with detections + IDs
   cv2.imshow("ZED Person Detection with YOLOv8, Depth & DeepSORT", image_np)


   # Exit on 'q' key press
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break


# Cleanup
zed.close()
cv2.destroyAllWindows()




