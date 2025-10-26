"""
advanced_cv_hub.py
A complete-ish advanced OpenCV demo:
- Motion detection (background subtraction)
- Face detection (Haar cascade)
- Simple centroid-based object tracker
- Trajectories drawing
- Snapshot and video recording
- CSV logging of events

Usage:
    python advanced_cv_hub.py --source 0
    python advanced_cv_hub.py --source path/to/video.mp4

Controls (when window focused):
    q : quit
    s : save snapshot image (snapshot_YYYYMMDD_HHMMSS.jpg)
    r : toggle recording (output video saved)
    d : toggle debug overlays (masks/boxes/ids)
"""
import cv2
import numpy as np
import argparse
import time
import datetime
import os
import csv
from collections import OrderedDict, deque
from math import hypot

# Optional pandas for nicer CSV export, fallback to csv module if not present
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False

# -------------------------
# Simple Centroid Tracker
# -------------------------
class CentroidTracker:
    """
    Tracks object centroids by assigning IDs and updating with nearest centroid matching.
    Not fancy (no Kalman), but works well for demos and slow movement.
    """
    def __init__(self, max_lost=30, max_distance=50):
        # next object ID to assign
        self.next_object_id = 0
        # maps object ID -> centroid (x, y)
        self.objects = OrderedDict()
        # maps object ID -> bounding box (startX, startY, endX, endY)
        self.bboxes = OrderedDict()
        # maps object ID -> number of consecutive frames it's not been seen
        self.lost = OrderedDict()
        # maximum frames to tolerate lost without removal
        self.max_lost = max_lost
        # max allowed distance for matching
        self.max_distance = max_distance
        # track trajectories (deque of recent centroids)
        self.trajectories = OrderedDict()

    def register(self, centroid, bbox):
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.lost[self.next_object_id] = 0
        self.trajectories[self.next_object_id] = deque(maxlen=64)
        self.trajectories[self.next_object_id].append(centroid)
        self.next_object_id += 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.bboxes:
            del self.bboxes[object_id]
        if object_id in self.lost:
            del self.lost[object_id]
        if object_id in self.trajectories:
            del self.trajectories[object_id]

    def update(self, detections):
        """
        detections: list of bounding boxes (startX, startY, endX, endY)
        returns: dict of object_id -> centroid
        """
        if len(detections) == 0:
            # increment lost counters and deregister if too lost
            to_deregister = []
            for obj_id in list(self.lost.keys()):
                self.lost[obj_id] += 1
                if self.lost[obj_id] > self.max_lost:
                    to_deregister.append(obj_id)
            for oid in to_deregister:
                self.deregister(oid)
            return self.objects

        # compute centroids for detections
        input_centroids = []
        input_bboxes = []
        for (startX, startY, endX, endY) in detections:
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids.append((cX, cY))
            input_bboxes.append((startX, startY, endX, endY))

        # if no existing objects, register all
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i])
            return self.objects

        # otherwise, build distance matrix between existing objects and new centroids
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=float)
        for i in range(len(object_centroids)):
            for j in range(len(input_centroids)):
                D[i, j] = hypot(object_centroids[i][0] - input_centroids[j][0],
                                object_centroids[i][1] - input_centroids[j][1])

        # find smallest value pairs
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.bboxes[object_id] = input_bboxes[col]
            self.lost[object_id] = 0
            self.trajectories[object_id].append(input_centroids[col])

            used_rows.add(row)
            used_cols.add(col)

        # find unmatched input centroids -> register
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)
        for col in unused_cols:
            self.register(input_centroids[col], input_bboxes[col])

        # find unmatched existing objects -> increment lost & deregister if needed
        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        for row in unused_rows:
            object_id = object_ids[row]
            self.lost[object_id] += 1
            if self.lost[object_id] > self.max_lost:
                self.deregister(object_id)

        return self.objects

# -------------------------
# Utility helpers
# -------------------------
def timestamp_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_snapshot(frame, out_dir="snapshots"):
    ensure_dir(out_dir)
    fname = f"snapshot_{timestamp_str()}.jpg"
    path = os.path.join(out_dir, fname)
    cv2.imwrite(path, frame)
    print(f"[INFO] Snapshot saved: {path}")
    return path

def write_csv_log(rows, header, outfile="events_log.csv"):
    if PANDAS_AVAILABLE:
        df = pd.DataFrame(rows, columns=header)
        df.to_csv(outfile, index=False)
    else:
        with open(outfile, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for r in rows:
                writer.writerow(r)
    print(f"[INFO] CSV log saved: {outfile}")

# -------------------------
# Detection utilities
# -------------------------
def find_contours_in_mask(mask, min_area=500):
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, x + w, y + h))
    return boxes

def detect_faces(gray_frame, face_cascade, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
    detections = face_cascade.detectMultiScale(gray_frame, scaleFactor=scaleFactor,
                                               minNeighbors=minNeighbors, minSize=minSize)
    boxes = []
    for (x, y, w, h) in detections:
        boxes.append((x, y, x + w, y + h))
    return boxes

# -------------------------
# Main app
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source", type=str, default="0",
                    help="Video source (0 for webcam or path to video file)")
    ap.add_argument("--min-area", type=int, default=600, help="Minimum contour area for motion")
    ap.add_argument("--display-width", type=int, default=1280, help="Display window width")
    ap.add_argument("--display-height", type=int, default=720, help="Display window height")
    ap.add_argument("--record-dir", type=str, default="recordings", help="Directory to save recordings")
    ap.add_argument("--snap-dir", type=str, default="snapshots", help="Directory to save snapshots")
    ap.add_argument("--log-file", type=str, default="events_log.csv", help="CSV log file")
    ap.add_argument("--show-mask", action="store_true", help="Show foreground mask overlay")
    args = ap.parse_args()

    src = args.source
    if src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("[ERROR] Cannot open video source:", args.source)
        return

    # prepare background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    # Haar cascade for face detection (bundled with opencv)
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(face_cascade_path):
        print("[WARN] Haar cascade file not found at expected location:", face_cascade_path)
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    tracker = CentroidTracker(max_lost=30, max_distance=80)

    # Recording & logging state
    recording = False
    video_writer = None
    out_dir = args.record_dir
    ensure_dir(out_dir)
    snap_dir = args.snap_dir
    ensure_dir(snap_dir)
    rows_log = []
    csv_header = ["timestamp", "type", "object_id", "xmin", "ymin", "xmax", "ymax", "centroid_x", "centroid_y", "info"]

    show_debug = True
    show_mask = args.show_mask

    # line for counting crossing events (horizontal center line)
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Couldn't read first frame from source.")
        return
    (H, W) = frame.shape[:2]
    display_w = args.display_width
    display_h = args.display_height
    scale_fx = display_w / float(W)
    scale_fy = display_h / float(H)
    # For counting, use a horizontal line at 2/3 height
    line_y = int(H * 2 / 3)
    count_in = 0
    count_out = 0

    print("[INFO] Starting main loop. Controls: q=quit, s=snapshot, r=toggle recording, d=toggle debug overlay, m=toggle mask view")
    fps_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream or cannot fetch frame.")
            break
        orig = frame.copy()
        frame_count += 1

        # resize for consistent processing performance
        # keep original for full res write, but process on smaller copy
        proc = cv2.resize(frame, (display_w, display_h))
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Background subtraction
        fgmask = backSub.apply(gray_blur)
        # Morphology to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)

        # threshold shadows (if any) — shadows are 127 in MOG2
        _, fgmask_bin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        # find motion boxes
        motion_boxes = find_contours_in_mask(fgmask_bin, min_area=args.min_area)

        # detect faces on the processed gray image
        face_boxes = detect_faces(gray_blur, face_cascade, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Combine motion and face detections into single detection set
        # Option: prefer face boxes (they're likely more important)
        all_boxes = []
        for b in motion_boxes:
            # scale down/up? motion boxes already in proc size
            all_boxes.append(b)
        for b in face_boxes:
            all_boxes.append(b)

        # update tracker with current detections
        objects = tracker.update(all_boxes)

        # draw center counting line
        cv2.line(proc, (0, line_y), (display_w, line_y), (255, 0, 255), 2)

        # draw detections and tracked IDs
        for oid, centroid in objects.items():
            bbox = tracker.bboxes.get(oid, None)
            if bbox is None:
                continue
            (sx, sy, ex, ey) = bbox
            # draw rectangle
            cv2.rectangle(proc, (sx, sy), (ex, ey), (0, 255, 0), 2)
            # draw ID
            cv2.putText(proc, f"ID {oid}", (sx, sy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # draw centroid
            cv2.circle(proc, centroid, 4, (0, 255, 255), -1)

            # check crossing the counting line: use trajectory points
            traj = tracker.trajectories.get(oid, None)
            if traj and len(traj) >= 2:
                # last two y coordinates
                y_prev = traj[-2][1]
                y_curr = traj[-1][1]
                # crossing upwards (from below to above)
                if y_prev > line_y and y_curr <= line_y:
                    count_out += 1
                    rows_log.append([timestamp_str(), "cross", oid, sx, sy, ex, ey, centroid[0], centroid[1], "out"])
                    print(f"[EVENT] ID {oid} crossed OUT (count_out={count_out})")
                # crossing downwards (from above to below)
                elif y_prev < line_y and y_curr >= line_y:
                    count_in += 1
                    rows_log.append([timestamp_str(), "cross", oid, sx, sy, ex, ey, centroid[0], centroid[1], "in"])
                    print(f"[EVENT] ID {oid} crossed IN (count_in={count_in})")

        # draw trajectories
        for oid, traj in tracker.trajectories.items():
            color = (int((oid * 37) % 255), int((oid * 57) % 255), int((oid * 97) % 255))
            pts = list(traj)
            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue
                thickness = int(np.sqrt(64 / float(i + 1)) * 2)
                cv2.line(proc, pts[i - 1], pts[i], color, thickness)

        # annotate counts
        cv2.putText(proc, f"In: {count_in}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(proc, f"Out: {count_out}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # show debug overlays optionally
        display = proc.copy()
        if show_debug and show_mask:
            # composite mask on the side for debug
            mask_bgr = cv2.cvtColor(fgmask_bin, cv2.COLOR_GRAY2BGR)
            # overlay a small mask in the top-right corner
            h, w = mask_bgr.shape[:2]
            # scale mask to 1/3 of display width
            scale = 0.25
            mh = int(display.shape[0] * scale)
            mw = int(display.shape[1] * scale)
            mask_small = cv2.resize(mask_bgr, (mw, mh))
            # place
            display[5:5 + mh, display.shape[1] - mw - 5:display.shape[1] - 5] = mask_small
            cv2.rectangle(display, (display.shape[1] - mw - 5, 5), (display.shape[1] - 5, 5 + mh), (255, 255, 255), 1)

        # display faces in separate color
        for (sx, sy, ex, ey) in face_boxes:
            cv2.rectangle(display, (sx, sy), (ex, ey), (255, 0, 0), 2)
            cv2.putText(display, "Face", (sx, sy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            rows_log.append([timestamp_str(), "face", "", sx, sy, ex, ey, int((sx + ex) / 2), int((sy + ey) / 2), "detected"])

        # show motion boxes in yellow
        for (sx, sy, ex, ey) in motion_boxes:
            cv2.rectangle(display, (sx, sy), (ex, ey), (0, 255, 255), 1)

        # small status bar at bottom
        fps = 1.0 / (time.time() - fps_time + 1e-6)
        fps_time = time.time()
        status = f"FPS: {fps:.1f} | Objects: {len(tracker.objects)} | Rec: {'ON' if recording else 'OFF'} | Debug: {'ON' if show_debug else 'OFF'}"
        cv2.putText(display, status, (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        # show the frame
        cv2.imshow("Advanced CV Hub", display)

        # handle recording
        if recording:
            # initialize writer if needed
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fname = f"record_{timestamp_str()}.mp4"
                outpath = os.path.join(out_dir, fname)
                video_writer = cv2.VideoWriter(outpath, fourcc, 20.0, (display.shape[1], display.shape[0]))
                print(f"[INFO] Recording started: {outpath}")
            video_writer.write(display)

        # keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Quitting.")
            break
        elif key == ord("s"):
            # save high-res snapshot of original frame
            snap_path = save_snapshot(orig, out_dir=snap_dir)
            rows_log.append([timestamp_str(), "snapshot", "", 0, 0, 0, 0, 0, 0, snap_path])
        elif key == ord("r"):
            recording = not recording
            if not recording and video_writer is not None:
                video_writer.release()
                video_writer = None
                print("[INFO] Recording stopped and saved.")
        elif key == ord("d"):
            show_debug = not show_debug
        elif key == ord("m"):
            show_mask = not show_mask

    # cleanup
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

    # write out log if we have rows
    if len(rows_log) > 0:
        write_csv_log(rows_log, csv_header, outfile=args.log_file)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
