import cv2
import torch
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime
from collections import defaultdict

# Config
allowed_classes = ["person", "chair"]
frame_skip = 3
buffer_seconds = 20
confidence_threshold = 0.6

# Initialize YOLOv8n model
model = YOLO("yolov8n.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open camera.")
    exit()

print("Press 's' to save buffer, 'q' to quit.")

frame_count = 0
frame_buffer = []
last_tracks = []
frame_times = []
fps = 0
tracker = DeepSort(max_age=60)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Failed to read frame.")
        break

    now = time.time()
    frame_times.append(now)
    frame_times = [t for t in frame_times if now - t < 1.0]
    fps = len(frame_times)

    frame_count += 1
    detections_to_track = []

    if frame_count % frame_skip == 0:
        results = model(frame, verbose=False)[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if label not in allowed_classes or conf < confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            detections_to_track.append(([x1, y1, x2, y2], conf, label))

        last_tracks = tracker.update_tracks(detections_to_track, frame=frame)

    # Draw tracks
    for track in last_tracks:
        if not track.is_confirmed():
            continue
        conf = getattr(track, 'det_conf', None)
        if conf is None or conf < confidence_threshold:
            continue

        l, t, r, b = map(int, track.to_tlbr())
        track_id = track.track_id
        track_class = getattr(track, 'det_class', 'unknown')

        cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0), 2)
        label_text = f"{track_class} ID:{track_id} {conf:.2f}"
        cv2.putText(frame, label_text, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Save to buffer
    frame_info = {
        "timestamp": now,
        "frame": frame.copy(),
        "detections": [
            (getattr(track, 'det_class', 'unknown'), track.track_id)
            for track in last_tracks
            if track.is_confirmed() and (conf := getattr(track, 'det_conf', None)) is not None and conf >= confidence_threshold
        ]
    }
    frame_buffer.append(frame_info)

    # Keep only buffer_seconds worth of frames
    frame_buffer = [f for f in frame_buffer if now - f["timestamp"] <= buffer_seconds]

    cv2.imshow("YOLOv8n + Deep SORT Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('s'):
        print("💾 Saving buffer to video, log, and summary...")
        if not frame_buffer:
            print("Buffer is empty.")
            continue

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"buffered_output_{timestamp_str}.mp4"
        log_filename = f"buffer_log_{timestamp_str}.txt"
        summary_filename = f"buffer_summary_{timestamp_str}.txt"

        height, width = frame_buffer[0]["frame"].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_fps = max(fps, 1)
        out_video = cv2.VideoWriter(video_filename, fourcc, output_fps, (width, height))

        unique_objects = set()
        class_counts = defaultdict(set)

        with open(log_filename, "w") as f_log:
            start_time = frame_buffer[0]["timestamp"]
            for item in frame_buffer:
                elapsed = item["timestamp"] - start_time
                det_str = ", ".join([f"{cls} ID:{tid}" for cls, tid in item["detections"]])
                f_log.write(f"Time {elapsed:.2f}s: {det_str}\n")
                for cls, tid in item["detections"]:
                    unique_objects.add((cls, tid))
                    class_counts[cls].add(tid)
                out_video.write(item["frame"])

        out_video.release()

        with open(summary_filename, "w") as f_summary:
            f_summary.write(f"Objects detected in {video_filename}:\n\n")
            for cls, tid in sorted(unique_objects, key=lambda x: (x[0], x[1])):
                f_summary.write(f"{cls} ID:{tid}\n")

            f_summary.write("\nTotal count per class:\n")
            total_objects = 0
            for cls, ids in class_counts.items():
                count = len(ids)
                total_objects += count
                f_summary.write(f"{cls}: {count}\n")

            f_summary.write(f"\n🔢 Total unique objects detected: {total_objects}\n")

        print(f"✅ Saved {video_filename}, {log_filename}, and {summary_filename}")

        # Reset state
        frame_buffer = []
        tracker = DeepSort(max_age=60)

cap.release()
cv2.destroyAllWindows()
