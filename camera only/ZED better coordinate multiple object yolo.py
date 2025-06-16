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

def exponential_moving_average(prev_value, new_value, alpha=0.9):
    if prev_value is None:
        return new_value
    if new_value is None:
        return prev_value
    return prev_value * (1 - alpha) + new_value * alpha


# Add to the previous function: Apply EMA to the distance as well
previous_distances = {}

previous_bboxes = {}

"""
def detect_and_track(frame, xyz_map, model, tracker, used_class_ids, last_write_time):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb_frame, verbose=False, classes=used_class_ids)

    deepsort_inputs = []
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

            confidence = confs[i].item()
            class_name = names[class_id]

            # Save original YOLO box (for red box)
            orig_x1, orig_y1, orig_x2, orig_y2 = boxes[i].astype(int)
            x1, y1, x2, y2 = orig_x1, orig_y1, orig_x2, orig_y2  # Will be replaced by mask box if valid

            # Override with segmentation mask bounding box if available
            if i < len(masks):
                mask = masks[i]
                mask_resized = cv2.resize(mask, (xyz_map.shape[1], xyz_map.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_bool = mask_resized.astype(bool)

                ys, xs = np.where(mask_bool)
                if len(xs) > 10 and len(ys) > 10:  # prevent bounding boxes on specks
                    x1 = int(np.min(xs))
                    y1 = int(np.min(ys))
                    x2 = int(np.max(xs))
                    y2 = int(np.max(ys))
                    pad = 5
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(frame.shape[1], x2 + pad)
                    y2 = min(frame.shape[0], y2 + pad)

            # --- Distance estimation ---
            distance_str = "N/A"
            distance = None

            # First, try using the mask-based method (median calculation)
            if i < len(masks):
                mask_resized = cv2.resize(masks[i], (xyz_map.shape[1], xyz_map.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
                mask_bool = mask_resized.astype(bool)
                points = xyz_map[mask_bool]
                if points.shape[0] > 0:
                    finite_mask = np.isfinite(points).all(axis=1)
                    valid_points = points[finite_mask]
                    if len(valid_points) > 0:
                        # Compute the mean point from valid points
                        mean_point = valid_points.mean(axis=0)
                        X, Y, Z = mean_point[:3]
                        distance = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

            # If no valid distance from the mask, fallback to depth ROI-based method
            if distance is None:
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

            # If no valid distance, fallback to center pixel method
            if distance is None:
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                if 0 <= center_x < xyz_map.shape[1] and 0 <= center_y < xyz_map.shape[0]:
                    point = xyz_map[center_y, center_x]
                    if np.isfinite(point[:3]).all():
                        X, Y, Z = point[:3]
                        distance = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

            # Force center point method if distance < 1m
            if distance is not None and distance < 1.5:
                #print("Forcing center point method due to close object distance.")
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                if 0 <= center_x < xyz_map.shape[1] and 0 <= center_y < xyz_map.shape[0]:
                    point = xyz_map[center_y, center_x]
                    if np.isfinite(point[:3]).all():
                        X, Y, Z = point[:3]
                        distance = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

            # If no valid distance is found
            if distance is None:
                distance_str = "N/A"
            else:
                # Apply smoothing (EMA) to the calculated distance
                track_id = int(ids[i])
                prev_distance = previous_distances.get(track_id, distance)
                smoothed_distance = exponential_moving_average(prev_distance, distance)
                previous_distances[track_id] = smoothed_distance
                distance_str = f"{smoothed_distance:.2f}m"

            track_id = int(ids[i])
            prev_box = previous_bboxes.get(track_id, (x1, y1, x2, y2))
            smoothed_x1 = exponential_moving_average(prev_box[0], x1)
            smoothed_y1 = exponential_moving_average(prev_box[1], y1)
            smoothed_x2 = exponential_moving_average(prev_box[2], x2)
            smoothed_y2 = exponential_moving_average(prev_box[3], y2)
            previous_bboxes[track_id] = (smoothed_x1, smoothed_y1, smoothed_x2, smoothed_y2)

            label = f"{class_name} ({confidence*100:.1f}%) {distance_str}"

            # Draw red YOLO box
            #cv2.rectangle(frame, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 0, 255), 2)

            # Draw green segmentation/mask box
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # average box blue
            x1_avg = int((orig_x1+x1)/2)
            x2_avg = int((orig_x2+x2)/2)
            y1_avg = int((orig_y1+y1)/2)
            y2_avg = int((orig_y2+y2)/2)
            cv2.rectangle(frame, (x1_avg, y1_avg), (x2_avg, y2_avg), (255, 0, 0), 2)

            # Draw label
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw mask overlay
            if i < len(masks):
                mask = masks[i]
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                colored_mask = np.zeros_like(frame)
                colored_mask[mask_resized.astype(bool)] = [255, 0, 0]
                frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)

            deepsort_inputs.append(([x1, y1, x2, y2], confidence, class_id))
            track_info.append((x1, y1, x2, y2, confidence, class_id, class_name, distance_str))

        # DeepSORT tracking
        tracks = tracker.update_tracks(deepsort_inputs, frame=frame)

        # CSV and TXT logging
        current_time = time.time()
        if current_time - last_write_time >= write_interval:
            last_write_time = current_time
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(csv_path, mode='a', newline='') as f_csv, open(txt_path, mode='a') as f_txt:
                writer = csv.writer(f_csv)
                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 150:
                        continue
                    tid = track.track_id
                    x1, y1, x2, y2 = map(int, track.to_ltrb())
                    for bx1, by1, bx2, by2, conf, cid, cname, dist_str in track_info:
                        if abs(x1 - bx1) < 20 and abs(y1 - by1) < 20:
                            writer.writerow([now, tid, cname, f"{conf:.2f}", dist_str])
                            f_txt.write(f"{now}\t{tid}\t{cname}\t{conf:.2f}\t{dist_str}\n")
                            break
                    cv2.putText(frame, f"ID: {tid}", (x1, y1 - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return frame, last_write_time
"""

def detect_and_track(frame, xyz_map, model, tracker, used_class_ids, last_write_time):
    global previous_distances, previous_bboxes

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb_frame, verbose=False, classes=used_class_ids)

    deepsort_inputs = []
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

            confidence = confs[i].item()
            class_name = names[class_id]

            orig_x1, orig_y1, orig_x2, orig_y2 = boxes[i].astype(int)
            x1, y1, x2, y2 = orig_x1, orig_y1, orig_x2, orig_y2

            if i < len(masks):
                mask = masks[i]
                mask_resized = cv2.resize(mask, (xyz_map.shape[1], xyz_map.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_bool = mask_resized.astype(bool)
                ys, xs = np.where(mask_bool)
                if len(xs) > 10 and len(ys) > 10:
                    x1 = max(0, int(np.min(xs)) - 5)
                    y1 = max(0, int(np.min(ys)) - 5)
                    x2 = min(frame.shape[1], int(np.max(xs)) + 5)
                    y2 = min(frame.shape[0], int(np.max(ys)) + 5)

            distance = None
            if i < len(masks):
                points = xyz_map[mask_bool]
                if points.shape[0] > 0:
                    valid_points = points[np.isfinite(points).all(axis=1)]
                    if len(valid_points) > 0:
                        distance = np.linalg.norm(valid_points.mean(axis=0)[:3])

            if distance is None:
                depth_roi = xyz_map[y1:y2, x1:x2]
                if depth_roi.size > 0:
                    valid = np.isfinite(depth_roi).all(axis=2)
                    points = depth_roi[valid]
                    if len(points) > 10:
                        distances = np.linalg.norm(points[:, :3], axis=1)
                        lower, upper = np.percentile(distances, [5, 95])
                        filtered = points[(distances >= lower) & (distances <= upper)]
                        if len(filtered) > 0:
                            distance = np.linalg.norm(filtered.mean(axis=0)[:3])

            if distance is None or distance < 1.2:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                if 0 <= cx < xyz_map.shape[1] and 0 <= cy < xyz_map.shape[0]:
                    point = xyz_map[cy, cx]
                    if np.isfinite(point[:3]).all():
                        distance = np.linalg.norm(point[:3])

            track_id = i  # temporary placeholder before DeepSORT gives real ID

            # EMA on distance
            prev_distance = previous_distances.get(track_id, None)
            smoothed_distance = exponential_moving_average(prev_distance, distance)
            previous_distances[track_id] = smoothed_distance
            distance_str = f"{smoothed_distance:.2f}m" if smoothed_distance else "N/A"

            # EMA on bounding box
            prev_box = previous_bboxes.get(track_id, (x1, y1, x2, y2))
            smoothed_box = (
                exponential_moving_average(prev_box[0], x1),
                exponential_moving_average(prev_box[1], y1),
                exponential_moving_average(prev_box[2], x2),
                exponential_moving_average(prev_box[3], y2),
            )
            previous_bboxes[track_id] = smoothed_box

            label = f"{class_name} ({confidence * 100:.1f}%) {distance_str}"

            x1_avg = int((orig_x1 + x1) / 2)
            y1_avg = int((orig_y1 + y1) / 2)
            x2_avg = int((orig_x2 + x2) / 2)
            y2_avg = int((orig_y2 + y2) / 2)
            cv2.rectangle(frame, (x1_avg, y1_avg), (x2_avg, y2_avg), (255, 0, 0), 2)

            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if i < len(masks):
                mask = masks[i]
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Ensure frame is 3 channels
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                colored_mask = np.zeros_like(frame)  # Now it matches frame exactly
                colored_mask[mask_resized.astype(bool)] = [255, 0, 0]

                frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)

            deepsort_inputs.append(([x1, y1, x2, y2], confidence, class_id))
            track_info.append((x1, y1, x2, y2, confidence, class_id, class_name, distance_str))

        tracks = tracker.update_tracks(deepsort_inputs, frame=frame)
        current_time = time.time()
        if current_time - last_write_time >= write_interval:
            last_write_time = current_time
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(csv_path, 'a', newline='') as f_csv, open(txt_path, 'a') as f_txt:
                writer = csv.writer(f_csv)
                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 150:
                        continue
                    tid = track.track_id
                    x1, y1, x2, y2 = map(int, track.to_ltrb())
                    for bx1, by1, bx2, by2, conf, cid, cname, dist_str in track_info:
                        if abs(x1 - bx1) < 20 and abs(y1 - by1) < 20:
                            writer.writerow([now, tid, cname, f"{conf:.2f}", dist_str])
                            f_txt.write(f"{now}\t{tid}\t{cname}\t{conf:.2f}\t{dist_str}\n")
                            break
                    cv2.putText(frame, f"ID: {tid}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return frame, last_write_time

def main():
    zed = initialize_zed_camera()
    model = load_yolo_model()
    tracker = initialize_tracker()
    used_class_ids = [0]  # Person only
    #used_class_ids = [56]  # Chair only
    #used_class_ids = [0, 56]  # Person & Chair

    image_zed = sl.Mat()
    xyz_zed = sl.Mat()

    last_write_time = time.time()  # Initialize the last write time
    start_time = time.time()       # Initialize the start time for the chronometer

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(xyz_zed, sl.MEASURE.XYZ)

            frame = image_zed.get_data()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            xyz_array = xyz_zed.get_data()
            xyz_array = np.array(xyz_array, dtype=np.float32)

            processed, last_write_time = detect_and_track(
                frame, xyz_array, model, tracker, used_class_ids, last_write_time
            )

            # --- Add Chronometer Overlay ---
            elapsed = time.time() - start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            millis = int((elapsed * 1000) % 1000)
            chrono_text = f"Time: {minutes:02}:{seconds:02}.{millis:03}"
            cv2.putText(
                processed, chrono_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
            )

            cv2.imshow("YOLOv8 + DeepSORT + ZED + 3D Distance", processed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()


try:
    main()
except KeyboardInterrupt:
    print("Interrupted. Exiting...")
