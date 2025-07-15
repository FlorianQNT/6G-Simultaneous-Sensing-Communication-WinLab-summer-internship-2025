import pyzed.sl as sl
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

# ============================
# Initialize ZED Camera
# ============================
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.METER
init_params.sdk_verbose = 1

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED Camera")
    exit(1)

# ============================
# Load YOLOv8 Model (Open Images V7)
# ============================
model = YOLO("yolov8l-oiv7.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# You can find every class name on ultralytics website under "Dataset YAML" part
# https://docs.ultralytics.com/datasets/detect/open-images-v7/#dataset-yaml

# Get class names
class_names = model.names  # dict: {id: 'class_name'}
used_class_labels = ["Chair", "Man"]
used_class_ids = [i for i, name in class_names.items() if name in used_class_labels]

# ============================
# Create containers
# ============================
objects = sl.Objects()
depth_map = sl.Mat()
previous_depths = {}
depth_history = {}
depth_smoothing_window = 8


def get_stable_depth(obj_id, new_depth):
    if obj_id not in depth_history:
        depth_history[obj_id] = [new_depth] * depth_smoothing_window
    depth_history[obj_id].append(new_depth)
    if len(depth_history[obj_id]) > depth_smoothing_window:
        depth_history[obj_id].pop(0)
    return np.median(depth_history[obj_id])


# ============================
# Initialize DeepSORT
# ============================
tracker = DeepSort(max_age=200, n_init=20, max_iou_distance=0.7)

# ============================
# Main Loop
# ============================
while zed.grab() == sl.ERROR_CODE.SUCCESS:
    image = sl.Mat()
    zed.retrieve_image(image, sl.VIEW.LEFT)
    image_np = image.get_data()
    zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
    depth_data = depth_map.get_data()

    if image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    results = model(image_np)
    deepsort_inputs = []

    if results:
        boxes = results[0].boxes
        detections = boxes.xywh.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(detections)):
            class_id = class_ids[i]
            if class_id not in used_class_ids:
                continue

            x_center, y_center, width, height = detections[i]
            confidence = confidences[i]
            class_name = class_names[class_id]

            # Convert to pixel center
            center_x = int(x_center)
            center_y = int(y_center)

            # Avoid out-of-bounds errors
            if center_y >= depth_data.shape[0] or center_x >= depth_data.shape[1]:
                continue

            raw_depth_value = depth_data[center_y, center_x]
            if np.isnan(raw_depth_value) or raw_depth_value <= 0:
                continue

            depth_value = get_stable_depth(class_id, raw_depth_value)

            # Threshold based on depth
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

            # Draw bounding box
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            color = (255, 0, 0) if movement == "Moving Towards" else (0, 0, 255) \
                if movement == "Moving Away" else (0, 255, 0)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 3)

            label = f"{class_name} ({confidence*100:.1f}%) | {movement} | Depth: {depth_value:.2f}m"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image_np, (x1, y1 - 25), (x1 + text_size[0], y1), (0, 0, 0), -1)
            cv2.putText(image_np, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Add to DeepSORT
            deepsort_inputs.append(([x1, y1, x2, y2], confidence, class_id))

    # Update tracker
    tracks = tracker.update_tracks(deepsort_inputs, frame=image_np)
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 150:
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.putText(image_np, f"ID: {track_id}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Show result
    cv2.imshow("ZED + YOLOv8 (OpenImages) + DeepSORT + Depth", image_np)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
zed.close()
cv2.destroyAllWindows()
