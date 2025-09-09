import time
from typing import Optional
import cv2
import numpy as np
import streamlit as st
import requests
import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

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
load_dotenv()

# -------------------- Simple Role Selection --------------------
st.sidebar.title("🔐 Login")
role = st.sidebar.selectbox("Select Role", ["Admin", "User"])

# Role check
is_admin = (role == "Admin")
role_display = "👑 Administrator" if is_admin else "👤 Operator"

# -------------------- Settings --------------------
st.sidebar.title(f"Settings ({role_display})")

# Backend selection
backend_choice = st.sidebar.selectbox("Detection Backend", ["Auto", "YOLO", "MediaPipe", "Path (.pt/.h5/.onnx)"])
model_path = st.sidebar.text_input("Model path or name", value="yolov8n.pt")

# Alert settings
threshold = st.sidebar.number_input("Alert threshold (people)", min_value=1, value=20)
st.sidebar.caption("⚠️ Alerts will be sent when crowd exceeds this number")

# Processing settings
frame_stride = st.sidebar.slider("Process every Nth frame", min_value=1, max_value=5, value=2)

# Video source
source_type = st.sidebar.selectbox("Video source", ["Webcam", "RTSP/HTTP URL", "Upload video file"])

# Backend integration
use_backend = st.sidebar.checkbox("Connect to Node backend", value=False)
backend_url = st.sidebar.text_input("Backend base URL", value="http://localhost:3000")

# Email settings (admin only)
if is_admin:
	st.sidebar.subheader("📧 Email Alert Settings")
	admin_email = st.sidebar.text_input("Admin email for alerts", value=os.getenv("ADMIN_EMAIL", ""))
	smtp_host = st.sidebar.text_input("SMTP Host", value=os.getenv("SMTP_HOST", ""))
	smtp_port = st.sidebar.number_input("SMTP Port", value=int(os.getenv("SMTP_PORT", "587")))
	smtp_user = st.sidebar.text_input("SMTP Username", value=os.getenv("SMTP_USER", ""))
	smtp_password = st.sidebar.text_input("SMTP Password", type="password", value=os.getenv("SMTP_PASSWORD", ""))
	from_email = st.sidebar.text_input("From Email", value=os.getenv("FROM_EMAIL", smtp_user))
else:
	# For users, use env vars or defaults
	admin_email = os.getenv("ADMIN_EMAIL", "")
	smtp_host = os.getenv("SMTP_HOST", "")
	smtp_port = int(os.getenv("SMTP_PORT", "587"))
	smtp_user = os.getenv("SMTP_USER", "")
	smtp_password = os.getenv("SMTP_PASSWORD", "")
	from_email = os.getenv("FROM_EMAIL", smtp_user)

# Video source specific settings
video_file = None
video_url: Optional[str] = None
webcam_index = 0
if source_type == "Webcam":
	webcam_index = st.sidebar.number_input("Webcam index", min_value=0, value=0)
	st.sidebar.caption("💡 If webcam fails, try index 1/2/3")
elif source_type == "RTSP/HTTP URL":
	video_url = st.sidebar.text_input("Stream URL (rtsp/http)")
elif source_type == "Upload video file":
	video_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

start = st.sidebar.button("🚀 Start Monitoring", type="primary")
stop_placeholder = st.sidebar.empty()

# Role info
with st.sidebar:
	st.divider()
	st.info(f"**Current Role:** {role_display}")

# Main UI
st.title("🏢 Crowd Safety Monitoring System")
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


def try_open_webcam(preferred_index: int) -> Optional[cv2.VideoCapture]:
	"""Try to open webcam using V4L2 first, then default backend; fall back over indices 0-3."""
	candidate_indices = [preferred_index] + [i for i in range(0, 4) if i != preferred_index]
	for idx in candidate_indices:
		cap = cv2.VideoCapture(int(idx), cv2.CAP_V4L2)
		if cap.isOpened():
			return cap
		cap.release()
		cap = cv2.VideoCapture(int(idx))
		if cap.isOpened():
			return cap
		cap.release()
	return None


def send_email_alert(count: int, threshold: int) -> bool:
	"""Send email alert when threshold is crossed."""
	try:
		if not (smtp_host and smtp_user and smtp_password and from_email and admin_email):
			return False
		
		subject = f"🚨 CROWD ALERT: Threshold Exceeded ({count} people)"
		body = f"""
Crowd Safety Alert

The crowd count has exceeded the safety threshold:

Current count: {count} people
Threshold: {threshold} people
Time: {time.strftime('%Y-%m-%d %H:%M:%S')}

Please take appropriate action immediately.

---
Crowd Safety Monitoring System
		""".strip()
		
		msg = EmailMessage()
		msg["Subject"] = subject
		msg["From"] = from_email
		msg["To"] = admin_email
		msg.set_content(body)
		
		with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
			server.starttls()
			server.login(smtp_user, smtp_password)
			server.send_message(msg)
		
		return True
	except Exception as e:
		st.error(f"Failed to send email: {e}")
		return False


if start:
	predictor = get_predictor(backend_choice, model_path)

	# Show predictor info
	with metrics_placeholder:
		st.subheader("🤖 Detection Model")
		if isinstance(predictor, DummyPredictor):
			st.error("Using dummy predictor. Try backend=Auto or YOLO.")
		elif isinstance(predictor, YoloPredictor):
			st.success("✅ YOLO person detection active")
		elif isinstance(predictor, KerasDensityPredictor):
			st.success("✅ Keras density model active")
		elif isinstance(predictor, MediaPipeFacePredictor):
			st.success("✅ MediaPipe face detection active")

	# On threshold change, push to backend as limit
	if use_backend:
		try:
			requests.post(f"{backend_url}/setLimit", json={"limit": int(threshold)}, timeout=2)
		except Exception:
			pass

	# Open video source
	if source_type == "Webcam":
		cap = try_open_webcam(int(webcam_index))
		if cap is None:
			devs = [d for d in os.listdir('/dev') if d.startswith('video')] if os.path.isdir('/dev') else []
			st.error(f"""
			📹 **Webcam not accessible**
			
			**Try:**
			1. Different webcam index (0-3)
			2. Check camera permissions
			3. Close other apps using camera
			4. On Linux: install v4l2loopback if needed
			
			**Detected devices:** {devs}
			""")
			st.stop()
	elif source_type == "RTSP/HTTP URL" and video_url:
		cap = cv2.VideoCapture(video_url)
	elif source_type == "Upload video file" and video_file is not None:
		import tempfile
		with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
			tmp.write(video_file.read())
			tmp_path = tmp.name
		cap = cv2.VideoCapture(tmp_path)
	else:
		st.error("Please provide a valid video source.")
		st.stop()

	if not cap or not cap.isOpened():
		st.error("Failed to open video source.")
		st.stop()

	count_text = "-"
	alert_active = False
	frame_count = 0
	run_flag = True
	last_alert_sent_ts: Optional[float] = None
	alert_cooldown_s = 60.0  # avoid spamming emails

	with metrics_placeholder:
		st.subheader("Metrics")
		count_slot = st.metric("Estimated people", count_text)
		alert_slot = st.empty()
		fps_slot = st.metric("FPS (approx)", "-")
		backend_status = st.empty()

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

		# Alert logic with email notification
		if count >= threshold and not alert_active:
			alert_active = True
			alert_msg = f"🚨 ALERT: Threshold exceeded! ({count} >= {threshold})"
			alert_slot.error(alert_msg)
			
			# Send email alert (with cooldown)
			if (last_alert_sent_ts is None) or (now - last_alert_sent_ts >= alert_cooldown_s):
				email_sent = send_email_alert(count, threshold)
				if email_sent:
					last_alert_sent_ts = now
					st.success(f"📧 Alert email sent to {admin_email}")
				else:
					st.warning("⚠️ Email not configured or failed to send")
		elif count < max(1, int(0.9 * threshold)) and alert_active:
			alert_active = False
			alert_slot.success("✅ Back to normal range")

		count_text = str(count)
		count_slot.metric("Estimated people", count_text)

		# If backend enabled, show status/alert from backend
		if use_backend:
			try:
				resp = requests.get(f"{backend_url}/status", timeout=2)
				js = resp.json()
				backend_status.info(f"Backend status: crowd={js.get('crowd')} status={js.get('status')}")
				# Optionally fetch alert message
				alert_resp = requests.get(f"{backend_url}/alert", timeout=2)
				alert_msg2 = alert_resp.json().get('message', '')
				if alert_msg2 and alert_active:
					alert_slot.warning(alert_msg2)
			except Exception:
				backend_status.warning("Backend unreachable")

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
