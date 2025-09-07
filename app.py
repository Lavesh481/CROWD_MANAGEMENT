import time
from typing import Optional
import cv2
import numpy as np
import streamlit as st

from model_loader import (
	load_predictor,
	load_predictor_auto,
	BasePredictor,
	DummyPredictor,
	YoloPredictor,
	KerasDensityPredictor,
	OnnxPredictor,
	MediaPipeFacePredictor,
)


st.set_page_config(page_title="Crowd Safety Monitoring", layout="wide")

# Sidebar controls
st.sidebar.title("Settings")
backend_choice = st.sidebar.selectbox("Backend", ["Auto", "YOLO", "MediaPipe", "Path (.pt/.h5/.onnx)"])
model_path = st.sidebar.text_input("Model path or name", value="yolov8n.pt")
threshold = st.sidebar.number_input("Alert threshold (people)", min_value=1, value=20)
frame_stride = st.sidebar.slider("Process every Nth frame", min_value=1, max_value=5, value=2)
source_type = st.sidebar.selectbox("Video source", ["Webcam", "RTSP/HTTP URL", "Upload video file"])

video_file = None
video_url: Optional[str] = None
webcam_index = 0
if source_type == "Webcam":
	webcam_index = st.sidebar.number_input("Webcam index", min_value=0, value=0)
elif source_type == "RTSP/HTTP URL":
	video_url = st.sidebar.text_input("Stream URL (rtsp/http)")
elif source_type == "Upload video file":
	video_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])  # type: ignore

start = st.sidebar.button("Start monitoring")
stop_placeholder = st.sidebar.empty()

# Main UI
st.title("Crowd Safety Monitoring")
col1, col2 = st.columns([2, 1])
frame_placeholder = col1.empty()
metrics_placeholder = col2.container()


@st.cache_resource(show_spinner=False)
def get_predictor(choice: str, path: Optional[str]) -> BasePredictor:
	choice_low = (choice or "").strip().lower()
	if choice_low == "auto":
		return load_predictor_auto()
	if choice_low == "mediapipe":
		return load_predictor("mediapipe")
	if choice_low == "yolo":
		return load_predictor(path or "yolov8n.pt")
	# Path mode or unknown -> delegate
	return load_predictor(path)


if start:
	predictor = get_predictor(backend_choice, model_path)

	# Show predictor info
	with metrics_placeholder:
		st.subheader("Model")
		if isinstance(predictor, DummyPredictor):
			st.error("Using dummy predictor. Try backend=Auto or YOLO, or type 'mediapipe'.")
		elif isinstance(predictor, YoloPredictor):
			st.info("Ultralytics YOLO active. Counting class 'person' (id 0) detections.")
		elif isinstance(predictor, KerasDensityPredictor):
			st.info("Keras density model active. Summing density map for counts.")
		elif isinstance(predictor, OnnxPredictor):
			st.warning("ONNX predictor is placeholder; adjust post-processing for your model.")
		elif isinstance(predictor, MediaPipeFacePredictor):
			st.info("MediaPipe Face Detection active. Counting visible faces.")

	# Open video source
	if source_type == "Webcam":
		cap = cv2.VideoCapture(int(webcam_index))
	elif source_type == "RTSP/HTTP URL" and video_url:
		cap = cv2.VideoCapture(video_url)
	elif source_type == "Upload video file" and video_file is not None:
		# Write uploaded file to temp buffer
		import tempfile
		with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
			tmp.write(video_file.read())
			tmp_path = tmp.name
		cap = cv2.VideoCapture(tmp_path)
	else:
		st.error("Please provide a valid video source.")
		st.stop()

	if not cap.isOpened():
		st.error("Failed to open video source.")
		st.stop()

	count_text = "-"
	alert_active = False
	frame_count = 0
	run_flag = True

	with metrics_placeholder:
		st.subheader("Metrics")
		count_slot = st.metric("Estimated people", count_text)
		alert_slot = st.empty()
		fps_slot = st.metric("FPS (approx)", "-")

	last_time = time.time()
	processed = 0

	while run_flag:
		ret, frame = cap.read()
		if not ret:
			break
		frame_count += 1
		if frame_count % frame_stride != 0:
			continue

		count, annotated = predictor.predict(frame)
		processed += 1

		# Compute approximate FPS
		now = time.time()
		dt = now - last_time
		if dt >= 1.0:
			fps_slot.metric("FPS (approx)", f"{processed/dt:.1f}")
			processed = 0
			last_time = now

		# Alert logic with simple hysteresis
		if count >= threshold and not alert_active:
			alert_active = True
			alert_slot.warning(f"ALERT: Threshold exceeded (count={count} >= {threshold})")
		elif count < max(1, int(0.9 * threshold)) and alert_active:
			alert_active = False
			alert_slot.info("Back to normal range")

		count_text = str(count)
		count_slot.metric("Estimated people", count_text)

		# Show frame
		show = annotated if annotated is not None else frame
		if show.ndim == 3 and show.shape[2] == 3:
			show_rgb = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
		else:
			show_rgb = show
		frame_placeholder.image(show_rgb, channels="RGB", use_column_width=True)

		# UI check for stop
		if stop_placeholder.button("Stop", key=f"stop_{frame_count}"):
			run_flag = False

	cap.release()
	st.success("Monitoring stopped.")
