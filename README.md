# SentinelCV 🎥🧠

**SentinelCV** is an advanced real-time computer vision hub built with OpenCV.  
It combines **motion detection**, **face detection**, **object tracking**, **trajectory visualization**, **event logging**, and **recording features** into a single, compact Python tool.

Ideal for:
- Surveillance demos
- Computer vision learning projects
- Object movement analysis
- Experimenting with tracking pipelines

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| **Motion Detection** | Uses background subtraction (MOG2) with noise cleanup |
| **Face Detection** | Haar Cascades (CPU-friendly, no GPU needed) |
| **Centroid Object Tracking** | Lightweight tracking without Kalman filter |
| **Trajectory Drawing** | Colored motion trails per tracked object |
| **Crossing Line Counting** | Detects directional movement (in/out) |
| **Snapshots** | Press **S** to save a full-resolution image |
| **Video Recording** | Press **R** to toggle recording as MP4 |
| **CSV Event Logging** | Stores detection & movement metadata |

---

## 🖥️ Demo Controls

| Key | Action |
|-----|--------|
| **q** | Quit |
| **s** | Save snapshot |
| **r** | Toggle video recording |
| **d** | Toggle debug overlays |
| **m** | Show/hide motion mask preview |

---

## 🧰 Installation

```bash
pip install opencv-python numpy pandas

▶️ Running the App
Use Webcam
python advanced_cv_hub.py --source 0

Use Video File
python advanced_cv_hub.py --source path/to/video.mp4

📦 Output Structure
SentinelCV/
 ├── snapshots/        # saved snapshots
 ├── recordings/       # recorded videos
 ├── events_log.csv    # CSV event logging
 └── advanced_cv_hub.py

🧠 How Tracking Works (Simplified Explanation)

Objects are detected → centroids are computed → nearest centroid matching assigns consistent IDs.
Trajectories are stored to allow:

Line crossing detection (In/Out counting)

Smooth path drawing

Object persistence during brief occlusions

No Kalman filters, no deep learning — simple and elegant.



```bash
pip install opencv-python numpy pandas
