"""
problem:
    -single object or object of different class, fluctuate by +-2cm
    -if 2 or more object of same class, bounding box in the middle between them
"""


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
write_interval = 0.125  # Time in seconds between each write


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
    # model = YOLO("yolov8l.pt")
    model = YOLO("yolov8l-seg.pt")
    # model = YOLO("yolo11l.pt")
    # model = YOLO("yolo11l-seg.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


def initialize_tracker():
    return DeepSort(max_age=200, n_init=20, max_iou_distance=0.7)


# Helper function to compute Exponential Moving Average (EMA)
def exponential_moving_average(prev_value, new_value, alpha=0.1):
    return prev_value * (1 - alpha) + new_value * alpha

# Add to the previous function: Apply EMA to the distance as well
previous_distances = {}

previous_bboxes = {}

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

            # ============================
            # Combining 3 Distance Methods
            # ============================
            distance_str = "N/A"
            distance = None

            # --- Segmentation Mask Distance ---
            if i < len(masks):
                mask = masks[i]
                mask_resized = cv2.resize(mask, (xyz_map.shape[1], xyz_map.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_bool = mask_resized.astype(bool)
                points = xyz_map[mask_bool]

                if points.shape[0] > 0:
                    finite_mask = np.isfinite(points).all(axis=1)
                    valid_points = points[finite_mask]
                    if len(valid_points) > 0:
                        mean_point = valid_points.mean(axis=0)
                        X, Y, Z = mean_point[:3]
                        distance = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
                        distance_str = f"{distance:.2f}m"

            # --- Bounding Box ROI Distance ---
            if distance is None:  # Only calculate if mask-based distance isn't found
                depth_roi = xyz_map[y1:y2, x1:x2]

                if depth_roi.size > 0:
                    valid = np.isfinite(depth_roi).all(axis=2)
                    points = depth_roi[valid]

                    if len(points) > 10:
                        distances = np.linalg.norm(points[:, :3], axis=1)
                        lower, upper = np.percentile(distances, [5, 95])
                        filtered_points = points[(distances >= lower) & (distances <= upper)]

                        if len(filtered_points) > 0:
                            mean_point = filtered_points.mean(axis=0)
                            X, Y, Z = mean_point[:3]
                            distance = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
                            distance_str = f"{distance:.2f}m"

            # --- Center Point Distance ---
            if distance is None:  # Only calculate if no other method has provided a distance
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                if 0 <= center_x < xyz_map.shape[1] and 0 <= center_y < xyz_map.shape[0]:
                    point = xyz_map[center_y, center_x]

                    if len(point) >= 3:
                        X, Y, Z = point[:3]
                        if np.isfinite(X) and np.isfinite(Y) and np.isfinite(Z):
                            distance = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
                            distance_str = f"{distance:.2f}m"

            # If no distance method works, leave as "N/A"
            if distance is None:
                distance_str = "N/A"

            # ============================
            # Apply Exponential Moving Average (EMA) for Distance Smoothing
            # ============================
            track_id = int(ids[i])

            # Initialize the previous distance if not present
            if track_id not in previous_distances:
                previous_distances[track_id] = distance

            prev_distance = previous_distances[track_id]

            # Apply EMA to smooth distance value
            smoothed_distance = exponential_moving_average(prev_distance, distance, alpha=0.1)  # Adjust alpha as needed

            # Update the previous distance with the new smoothed value
            previous_distances[track_id] = smoothed_distance

            # Update the distance string to show smoothed value
            distance_str = f"{smoothed_distance:.2f}m"

            # ============================
            # Apply EMA to Bounding Box Coordinates (same as before)
            # ============================
            prev_x1, prev_y1, prev_x2, prev_y2 = previous_bboxes.get(track_id, (x1, y1, x2, y2))
            # Default to current coordinates if track_id is new

            smoothed_x1 = exponential_moving_average(prev_x1, x1)
            smoothed_y1 = exponential_moving_average(prev_y1, y1)
            smoothed_x2 = exponential_moving_average(prev_x2, x2)
            smoothed_y2 = exponential_moving_average(prev_y2, y2)

            # Update the previous bounding box with the new smoothed values
            previous_bboxes[track_id] = (smoothed_x1, smoothed_y1, smoothed_x2, smoothed_y2)

            # ============================
            # Draw bounding box and label
            # ============================
            label = f"{class_name} ({confidence * 100:.1f}%) {distance_str}"
            cv2.rectangle(frame, (int(smoothed_x1), int(smoothed_y1)), (int(smoothed_x2), int(smoothed_y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(smoothed_x1), int(smoothed_y1) - 10),
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
    #used_class_ids = [0, 56]

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
