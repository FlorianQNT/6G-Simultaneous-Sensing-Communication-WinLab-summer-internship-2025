import cv2
import torch
from collections import Counter
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pyzed.sl as sl
import numpy as np
import os
import csv
from datetime import datetime
import time

log_dir = r"C:\Users\native\Desktop\distance accuracy"
os.makedirs(log_dir, exist_ok=True)
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(log_dir, f"detections_{timestamp_str}.csv")
txt_path = os.path.join(log_dir, f"detections_{timestamp_str}.txt")

# Initialize files with headers
with open(csv_path, mode='w', newline='') as f_csv, open(txt_path, mode='w') as f_txt:
    writer = csv.writer(f_csv)
    writer.writerow(["Timestamp", "ID", "Class", "Confidence", "Distance (m)"])  # CSV Header
    f_txt.write("Timestamp\tID\tClass\tConfidence\tDistance (m)\n")  # TXT Header

# Control the writing frequency
write_interval = 0.25  # Time in seconds between each write


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
    #model = YOLO("yolov8l.pt")
    model = YOLO("yolov8l-seg.pt")
    #model = YOLO("yolo11l.pt")
    #model = YOLO("yolo11l-seg.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


def initialize_tracker():
    return DeepSort(max_age=200, n_init=20, max_iou_distance=0.7)


def detect_and_track(frame, xyz_map, model, tracker, used_class_ids, last_write_time):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb_frame, verbose=False, classes=used_class_ids)

    deepsort_inputs = []
    class_count = Counter()
    track_info = []

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
            class_count[class_name] += 1
            """
            test with 1 chair at the same place for the 3 way to get the distance
            yolov8l:
            for center point: 2.31 to 2.50m => middle at 2.40m 
            for segmentation mask: not usable without segmentation
            for valid 3D point inside bounding box: 2.80 to 3.84m (spike at 5.57 and 5.63m) => middle at 3.52m 

            yolov8l-seg:
            for center point: 2.35 to 2.57m => middle at 2.46m
            for segmentation mask: 2.70 to 3.02m => middle at 2.86m
            for valid 3D point inside bounding box: 2.86 to 3.48m  => middle at 3.17m 

            yolov11l: 
            for center point: 2.38 to 2.55m => middle at 2.47m
            for segmentation mask: not usable without segmentation
            for valid 3D point inside bounding box: 2.84 to 3.49m (spike at 3.85m) => middle at 3.16m

            yolov11l-seg: 
            for center point: 2.35 to 2.58m => middle at 2.47m
            for segmentation mask: 2.68 to 2.92m => middle at 2.80m
            for valid 3D point inside bounding box: 2.85 to 3.59m (spike at 4.95 and 5.02m) => middle at 3.22m
            
            =========================================================
            
            test with 2 chair at the same place for the 3 way to get the distance
            yolov8l:
            for center point: - chair1: 2.42 to 2.88m (with 12 N/A in 16s) => middle at 2.65m
                              - chair2: 2.27 to 2.73m => middle at 2.5m
            for segmentation mask: not usable without segmentation
            for valid 3D point inside bounding box: - chair1: 2.51 to 3.01m => middle at 2.76m
                                                    - chair2: 2.66 to 3.33m (spike at 4.29m) => middle at 2.995m

            yolov8l-seg:
            for center point: - chair1: 2.17 to 2.52m  => middle at 2.345m
                              - chair2: 2.31 to 2.88m (with 7 N/A in 16s) => middle at 2.595m
            for segmentation mask: - chair1: 2.45 to 2.64m => middle at 2.545m
                                   - chair2: 2.64 to 2.72m (spike at 2.99m) => middle at 2.68m
            for valid 3D point inside bounding box: - chair1: 2.63 to 3.19m => middle at 2.91m
                                                    - chair2: 2.65 to 3.13m (spike at 3.40, 3.47, 4.57 and 5.48m) 
                                                            => middle at 2.89m

            yolov11l: 
            for center point: - chair1: 2.41 to 2.88m (with 8 N/A in 16s) => middle at 2.645m
                              - chair2: 2.31 to 2.59m => middle at 2.45m
            for segmentation mask: not usable without segmentation
            for valid 3D point inside bounding box: - chair1: 2.72 to 3.84m => middle at 3.28m
                                                    - chair2: 2.37 to 3.93m (spike at 4.13 and 4.38m) 
                                                            => middle at 3.15m

            yolov11l-seg: 
            for center point: - chair1: 2.41 to 2.84m (with 7 N/A in 16s) => middle at 2.625m
                              - chair2: 2.08 to 2.48m (with 4 N/A in 16s) => middle at 2.28m
            for segmentation mask: - chair1: 2.48 to 2.78m => middle at 2.63m
                                   - chair2: 2.60 to 2.98m => middle at 2.79m
            for valid 3D point inside bounding box: - chair1: 2.71 to 3.55m => middle at 3.13m
                                                    - chair2: 2.47 to 3.55m (spike at 4.37m) => middle at 3.01m
            """
            # ============================
            # How to calculate the distance
            # ============================
            """
            # --- Distance using XYZ coordinates (center point method) ---
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            if 0 <= center_x < xyz_map.shape[1] and 0 <= center_y < xyz_map.shape[0]:
                point = xyz_map[center_y, center_x]

                if len(point) >= 3:
                    X, Y, Z = point[:3]  # Use only the first three components
                    if np.isfinite(X) and np.isfinite(Y) and np.isfinite(Z):
                        distance = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
                        distance_str = f"{distance:.2f}m"
                    else:
                        distance_str = "N/A"
                else:
                    distance_str = "N/A"
            else:
                distance_str = "N/A"
            
            # --- Alternative: Distance using segmentation mask (average over valid 3D points) --- 
            if i < len(masks):
                mask = masks[i]
                mask_resized = cv2.resize(mask, (xyz_map.shape[1], xyz_map.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_bool = mask_resized.astype(bool)

                points = xyz_map[mask_bool]  # (N, C)
                if points.shape[0] > 0:
                    finite_mask = np.isfinite(points).all(axis=1)
                    valid_points = points[finite_mask]
                    if len(valid_points) > 0:
                        mean_point = valid_points.mean(axis=0)
                        X, Y, Z = mean_point[:3]  # Use only X, Y, Z
                        distance = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
                        distance_str = f"{distance:.2f}m"
                    else:
                        distance_str = "N/A"
                else:
                    distance_str = "N/A"
            """
            # --- New: Distance using average over valid 3D points inside bounding box ---
            depth_roi = xyz_map[y1:y2, x1:x2]

            if depth_roi.size > 0:
                valid = np.isfinite(depth_roi).all(axis=2)
                points = depth_roi[valid]

                if len(points) > 10:
                    # Remove extreme depth outliers (e.g., keep middle 90%)
                    distances = np.linalg.norm(points[:, :3], axis=1)
                    lower, upper = np.percentile(distances, [5, 95])
                    filtered_points = points[(distances >= lower) & (distances <= upper)]

                    if len(filtered_points) > 0:
                        mean_point = filtered_points.mean(axis=0)
                        X, Y, Z = mean_point[:3]
                        distance = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
                        distance_str = f"{distance:.2f}m"
                    else:
                        distance_str = "N/A"
                else:
                    distance_str = "N/A"
            else:
                distance_str = "N/A"

            # Draw bounding box and label with class and distance
            label = f"{class_name} ({confidence * 100:.1f}%) {distance_str}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw segmentation mask (optional)
            if i < len(masks):
                mask = masks[i]
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                colored_mask = np.zeros_like(frame)
                colored_mask[mask_resized.astype(bool)] = [255, 0, 0]
                frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)

            deepsort_inputs.append(([x1, y1, x2, y2], confidence, class_id))
            track_info.append((x1, y1, x2, y2, confidence, class_id, class_name, distance_str))

        # Update DeepSORT tracks
        tracks = tracker.update_tracks(deepsort_inputs, frame=frame)

        # Control the writing frequency
        current_time = time.time()
        if current_time - last_write_time >= write_interval:
            last_write_time = current_time  # Update the last write time

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(csv_path, mode='a', newline='') as f_csv, open(txt_path, mode='a') as f_txt:
                writer = csv.writer(f_csv)

                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 150:
                        continue

                    tid = track.track_id
                    x1, y1, x2, y2 = map(int, track.to_ltrb())
                    # Match bounding box with stored info
                    for bx1, by1, bx2, by2, conf, cid, cname, dist_str in track_info:
                        if abs(x1 - bx1) < 20 and abs(y1 - by1) < 20:
                            writer.writerow([now, tid, cname, f"{conf:.2f}", dist_str])
                            f_txt.write(f"{now}\t{tid}\t{cname}\t{conf:.2f}\t{dist_str}\n")
                            break

                    cv2.putText(frame, f"ID: {tid}", (x1, y1 - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return frame, last_write_time


def main():
    zed = initialize_zed_camera()
    model = load_yolo_model()
    tracker = initialize_tracker()
    #used_class_ids = [0]  # Person only
    used_class_ids = [56]  # Chair only

    image_zed = sl.Mat()
    xyz_zed = sl.Mat()

    last_write_time = time.time()  # Initialize the last write time

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(xyz_zed, sl.MEASURE.XYZ)

            frame = image_zed.get_data()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            xyz_array = xyz_zed.get_data()
            xyz_array = np.array(xyz_array, dtype=np.float32)

            processed, last_write_time = detect_and_track(frame, xyz_array, model, tracker, used_class_ids,
                                                          last_write_time)
            cv2.imshow("YOLOv8 + DeepSORT + ZED + 3D Distance", processed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()


try:
    main()
except KeyboardInterrupt:
    print("Interrupted. Exiting...")
