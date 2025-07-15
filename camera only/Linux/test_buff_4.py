import cv2
import torch
import time
import os
import shutil
import zipfile
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Configuration
allowed_classes = ["person", "chair"]
frame_skip = 3
buffer_seconds = 20
confidence_threshold = 0.6
save_interval = 20
archive_threshold = 120  # 2 minutes
max_saved_folders = 3

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def create_tracker():
    return DeepSort(max_age=60)

tracker = create_tracker()

# Paths
OUTPUT_ROOT = "output"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Session folder
date_str = datetime.now().strftime("%Y%m%d")
start_time_str = datetime.now().strftime("%H-%M-%S")

session_folder = os.path.join(OUTPUT_ROOT, f"session_{date_str}")
run_folder = os.path.join(session_folder, start_time_str)
archive_folder = os.path.join(run_folder, "compressed_archives")

os.makedirs(run_folder, exist_ok=True)
os.makedirs(archive_folder, exist_ok=True)

# Move old files to local compressed_archives folder
def move_old_files_to_archive(run_path, archive_path):
    now = time.time()
    for fname in os.listdir(run_path):
        fpath = os.path.join(run_path, fname)
        if os.path.isfile(fpath):
            ftime = os.path.getmtime(fpath)
            if now - ftime > archive_threshold:
                shutil.move(fpath, os.path.join(archive_path, fname))

# Clean old sessions
def cleanup_old_sessions(base_folder, max_folders):
    subdirs = sorted([d for d in os.listdir(base_folder) if d.startswith("session_") and os.path.isdir(os.path.join(base_folder, d))])
    if len(subdirs) <= max_folders:
        return
    to_delete = subdirs[:-max_folders]
    for folder in to_delete:
        shutil.rmtree(os.path.join(base_folder, folder))

# Compress previous sessions (before today)
def compress_old_sessions(base_folder):
    today_str = datetime.now().strftime("%Y%m%d")
    for session_name in os.listdir(base_folder):
        session_path = os.path.join(base_folder, session_name)
        if not session_name.startswith("session_") or not os.path.isdir(session_path):
            continue
        session_date = session_name.replace("session_", "")
        if session_date < today_str:
            zip_path = os.path.join(base_folder, f"{session_name}.zip")
            if os.path.exists(zip_path):
                print(f"📦 Already compressed: {zip_path}")
                continue
            print(f"🗜️ Compressing old session: {session_path}")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(session_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        arcname = os.path.relpath(full_path, base_folder)
                        zipf.write(full_path, arcname)
            shutil.rmtree(session_path)
            print(f"✅ Compressed and removed: {session_path}")

# Video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open camera.")
    exit()

frame_count = 0
frame_buffer = []
last_tracks = []
frame_times = []
fps = 0
last_save_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
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

    # Save buffer
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
    frame_buffer = [f for f in frame_buffer if now - f["timestamp"] <= buffer_seconds]

    if now - last_save_time >= save_interval and frame_buffer:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join(run_folder, f"buffered_output_{timestamp_str}.mp4")
        log_filename = os.path.join(run_folder, f"buffer_log_{timestamp_str}.txt")
        summary_filename = os.path.join(run_folder, f"buffer_summary_{timestamp_str}.txt")

        height, width = frame_buffer[0]["frame"].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(video_filename, fourcc, max(fps, 1), (width, height))

        summary_dict = {}
        with open(log_filename, "w") as log_file:
            start_time = frame_buffer[0]["timestamp"]
            for item in frame_buffer:
                elapsed = item["timestamp"] - start_time
                det_str = ", ".join([f"{cls} ID:{tid}" for cls, tid in item["detections"]])
                log_file.write(f"Time {elapsed:.2f}s: {det_str}\n")
                for cls, tid in item["detections"]:
                    summary_dict.setdefault(cls, set()).add(tid)
                out_video.write(item["frame"])
        out_video.release()

        with open(summary_filename, "w") as summary_file:
            total = 0
            for cls, ids in summary_dict.items():
                id_list = sorted(ids)
                summary_file.write(f"{cls}: {len(id_list)} unique ID(s): {id_list}\n")
                total += len(id_list)
            summary_file.write(f"\n🔢 Total unique objects: {total}\n")

        print(f"✅ Saved files to {run_folder}")
        last_save_time = now
        frame_buffer.clear()
        tracker = create_tracker()

        move_old_files_to_archive(run_folder, archive_folder)
        cleanup_old_sessions(OUTPUT_ROOT, max_saved_folders)

    cv2.imshow("YOLOv8 + Deep SORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# On exit: compress local compressed_archives folder
if os.path.exists(archive_folder) and os.listdir(archive_folder):
    zip_path = os.path.join(run_folder, f"archives_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
    print(f"🗜️ Compressing archive folder to: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(archive_folder):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, archive_folder)
                zipf.write(full_path, arcname)
    shutil.rmtree(archive_folder)
    print("✅ Archive folder compressed and removed.")
else:
    print("ℹ️ No old files to archive in this session.")

# ➕ Compress full old session folders from previous days
compress_old_sessions(OUTPUT_ROOT)
