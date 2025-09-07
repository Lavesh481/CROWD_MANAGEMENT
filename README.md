# Crowd Safety Management (Minimal Scaffold)

This Streamlit app provides real-time counting using multiple backends:
- Auto: YOLO (yolov8n.pt auto-download) with fallback to MediaPipe Face Detection
- YOLO: Ultralytics person detection (counts class `person`)
- MediaPipe: counts visible faces (fast, lightweight)
- Path: load a specific model file or name (.pt/.h5/.onnx) or 'mediapipe'

## Quick start

```bash
python3 -m venv /home/lavesh/CROWD_MANAGEMENT/.venv
/home/lavesh/CROWD_MANAGEMENT/.venv/bin/python -m pip install --upgrade pip
/home/lavesh/CROWD_MANAGEMENT/.venv/bin/pip install -r /home/lavesh/CROWD_MANAGEMENT/requirements.txt
/home/lavesh/CROWD_MANAGEMENT/.venv/bin/streamlit run /home/lavesh/CROWD_MANAGEMENT/app.py
```

In the sidebar:
- Backend: choose Auto (recommended), YOLO, MediaPipe, or Path
- Model path or name:
  - Auto/YOLO: leave as `yolov8n.pt` to auto-download
  - MediaPipe: not used
  - Path: enter a valid path or keyword (e.g., `mediapipe`, `yolov8s.pt`)
- Choose your video source and click Start

### Deploy to Streamlit Cloud
- Main file: `app.py`
- Python version: 3.12
- Run command (auto): `streamlit run app.py`

### Notes
- YOLO provides better person counting; MediaPipe is a quick face-based fallback.
- If you later want to use a Keras `.h5` crowd model, select "Path" and point to it (custom layers may be required).
# CROWD_MANAGEMENT1
