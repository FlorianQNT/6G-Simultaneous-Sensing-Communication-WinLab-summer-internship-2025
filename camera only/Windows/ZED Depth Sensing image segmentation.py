import cv2
import torch
import time
#from collections import Counter
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pyzed.sl as sl
import numpy as np
import os
from datetime import datetime

# === CONFIG OUTPUT FILE ===
log_dir = r"C:\Users\native\Desktop\distance accuracy"
os.makedirs(log_dir, exist_ok=True)

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_TXT = os.path.join(log_dir, f"detections_{timestamp_str}.txt")
OUTPUT_CSV = os.path.join(log_dir, f"detections_{timestamp_str}.csv")

# === HEADER OUTPUT FILE ===
for output_file in [OUTPUT_TXT, OUTPUT_CSV]:
    with open(output_file, "w") as f:
        f.write("Timestamp,Class,Confidence,Distance(m),ID,Movement\n")

previous_depths = {}

def get_stable_depth(class_id, raw_depth):
    return raw_depth

def initialize_zed_camera():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera:", status)
        exit()
    return zed

def load_yolo_model():
    model = YOLO("yolov8l-seg.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

def initialize_tracker():
    return DeepSort(max_age=200, n_init=20, max_iou_distance=0.7)

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2b - x1b) * (y2b - y1b)
    return inter_area / float(box1_area + box2_area - inter_area + 1e-6)

def detect_and_track(frame, depth_map, model, tracker, used_class_ids, last_write_time, object_ids):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb_frame, verbose=False, classes=used_class_ids)

    deepsort_inputs = []
    current_time = time.time()
    write_allowed = current_time - last_write_time >= 0.25

    if results and results[0].boxes:
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        ids = result.boxes.cls.cpu().numpy().astype(int)
        names = model.names
        masks = result.masks.data.cpu().numpy() if result.masks else []

        for i, class_id in enumerate(ids):
            if class_id not in used_class_ids:
                continue

            x1, y1, x2, y2 = boxes[i].astype(int)
            confidence = confs[i].item()
            class_name = names[class_id]

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            if 0 <= center_x < depth_map.shape[1] and 0 <= center_y < depth_map.shape[0]:
                raw_depth = depth_map[center_y, center_x]
                distance = get_stable_depth(class_id, raw_depth)
                distance_str = f"{distance:.2f}" if np.isfinite(distance) else "N/A"
            else:
                distance = float('inf')
                distance_str = "N/A"

            if distance < 1.0:
                movement_threshold = 0.002
            elif distance < 1.5:
                movement_threshold = 0.0035
            elif distance < 2:
                movement_threshold = 0.0043
            elif distance < 3:
                movement_threshold = 0.0054
            else:
                movement_threshold = 0.07

            if class_id not in previous_depths:
                previous_depths[class_id] = distance

            depth_change = abs(distance - previous_depths[class_id])
            if depth_change > movement_threshold:
                movement = "Moving Towards" if distance < previous_depths[class_id] else "Moving Away"
            elif depth_change < movement_threshold / 2:
                movement = "Stationary"
            else:
                movement = "Uncertain"

            previous_depths[class_id] = distance

            if movement == "Moving Towards":
                color = (255, 0, 0)
            elif movement == "Moving Away":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            object_id = None
            for existing_id, bbox in object_ids.items():
                if iou(bbox, [x1, y1, x2, y2]) > 0.3:
                    object_id = existing_id
                    break
            if object_id is None:
                object_id = len(object_ids) + 1
            object_ids[object_id] = [x1, y1, x2, y2]

            if write_allowed:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                line = f"{timestamp},{class_name},{confidence:.2f},{distance_str},{object_id},{movement}\n"
                with open(OUTPUT_TXT, "a") as f_txt, open(OUTPUT_CSV, "a") as f_csv:
                    f_txt.write(line)
                    f_csv.write(line)

            label = f"{class_name} ({confidence*100:.1f}%) {distance_str}m, ID: {object_id}, {movement}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if i < len(masks):
                mask = masks[i]
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                colored_mask = np.zeros_like(frame)
                colored_mask[mask_resized.astype(bool)] = color
                frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)

            deepsort_inputs.append(([x1, y1, x2, y2], confidence, class_id))

    return frame, (current_time if write_allowed else last_write_time), object_ids

def main():
    zed = initialize_zed_camera()
    model = load_yolo_model()
    tracker = initialize_tracker()
    used_class_ids = [56]  # chair

    image_zed = sl.Mat()
    depth_zed = sl.Mat()

    last_write_time = time.time()
    object_ids = {}

    print(f"output file :\n TXT: {OUTPUT_TXT}\n CSV: {OUTPUT_CSV}")

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)

            frame = image_zed.get_data()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            depth_array = np.array(depth_zed.get_data(), dtype=np.float32)

            processed, last_write_time, object_ids = detect_and_track(
                frame, depth_array, model, tracker, used_class_ids, last_write_time, object_ids
            )

            cv2.imshow("YOLOv8 + DeepSORT + ZED", processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()

try:
    main()
except KeyboardInterrupt:
    print("Interrupted. Exiting...")
