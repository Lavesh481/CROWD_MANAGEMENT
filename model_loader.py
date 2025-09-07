import os
from typing import Optional, Tuple
import numpy as np

# Optional imports guarded at runtime
try:
	from ultralytics import YOLO  # type: ignore
except Exception:
	YOLO = None  # type: ignore

try:
	import onnxruntime as ort  # type: ignore
except Exception:
	ort = None  # type: ignore

try:
	from tensorflow.keras.models import load_model  # type: ignore
except Exception:
	load_model = None  # type: ignore

try:
	import mediapipe as mp  # type: ignore
except Exception:
	mp = None  # type: ignore


class BasePredictor:
	"""Abstract predictor interface to normalize model inference."""

	def predict(self, frame_bgr: np.ndarray) -> Tuple[int, Optional[np.ndarray]]:
		"""
		Returns:
			count: estimated number of people
			annotated: optional annotated frame (BGR)
		"""
		raise NotImplementedError


class DummyPredictor(BasePredictor):
	def predict(self, frame_bgr: np.ndarray) -> Tuple[int, Optional[np.ndarray]]:
		return 0, frame_bgr


class YoloPredictor(BasePredictor):
	def __init__(self, weights_path: str = "yolov8n.pt"):
		if YOLO is None:
			raise RuntimeError("Ultralytics YOLO is not installed")
		self.model = YOLO(weights_path)

	def predict(self, frame_bgr: np.ndarray) -> Tuple[int, Optional[np.ndarray]]:
		# Ultralytics expects RGB
		rgb = frame_bgr[:, :, ::-1]
		results = self.model(rgb, verbose=False)
		count = 0
		annotated = frame_bgr.copy()
		try:
			res = results[0]
			if hasattr(res, 'boxes') and res.boxes is not None:
				# person class id is commonly 0 in COCO
				cls = res.boxes.cls.cpu().numpy().astype(int)
				count = int((cls == 0).sum())
				# Draw boxes
				if hasattr(res, 'plot'):
					annotated = res.plot()
		except Exception:
			pass
		return count, annotated


class KerasDensityPredictor(BasePredictor):
	def __init__(self, h5_path: str):
		if load_model is None:
			raise RuntimeError("TensorFlow/Keras is not installed")
		self.model = load_model(h5_path)

	def predict(self, frame_bgr: np.ndarray) -> Tuple[int, Optional[np.ndarray]]:
		# Heuristic: resize to model input, assume output is density map, sum => count
		import cv2  # local import to avoid global hard dep in other paths
		input_shape = self.model.inputs[0].shape
		h, w = int(input_shape[1]), int(input_shape[2])
		rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
		resized = cv2.resize(rgb, (w, h))
		norm = resized.astype(np.float32) / 255.0
		batched = np.expand_dims(norm, axis=0)
		pred = self.model.predict(batched, verbose=0)
		# If density map, sum; if scalar, cast
		if pred.ndim >= 2:
			count = float(np.maximum(pred, 0).sum())
		else:
			count = float(pred.ravel()[0])
		return int(round(count)), frame_bgr


class OnnxPredictor(BasePredictor):
	def __init__(self, onnx_path: str):
		if ort is None:
			raise RuntimeError("onnxruntime is not installed")
		self.sess = ort.InferenceSession(onnx_path)  # pragma: no cover
		self.input_name = self.sess.get_inputs()[0].name

	def predict(self, frame_bgr: np.ndarray) -> Tuple[int, Optional[np.ndarray]]:
		# Placeholder: model-specific post-processing is needed; return dummy until configured
		return 0, frame_bgr


class MediaPipeFacePredictor(BasePredictor):
	def __init__(self):
		if mp is None:
			raise RuntimeError("mediapipe is not installed")
		self.solution = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

	def predict(self, frame_bgr: np.ndarray) -> Tuple[int, Optional[np.ndarray]]:
		import cv2
		rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
		res = self.solution.process(rgb)
		count = 0
		annotated = frame_bgr.copy()
		if res.detections:
			count = len(res.detections)
			ih, iw = annotated.shape[:2]
			for det in res.detections:
				bbox = det.location_data.relative_bounding_box
				x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
				cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
		return count, annotated


def load_predictor_auto() -> BasePredictor:
	"""Auto select the best available backend: YOLO -> MediaPipe -> Dummy."""
	# Try YOLO (auto-download yolov8n.pt)
	try:
		if YOLO is not None:
			return YoloPredictor("yolov8n.pt")
	except Exception:
		pass
	# Try MediaPipe faces
	try:
		return MediaPipeFacePredictor()
	except Exception:
		pass
	# Fallback
	return DummyPredictor()


def load_predictor(model_path: Optional[str]) -> BasePredictor:
	"""Load predictor from explicit model path or keyword; if 'auto', use auto resolver."""
	if not model_path or model_path.strip().lower() in {"auto", "default"}:
		return load_predictor_auto()

	ext = os.path.splitext(model_path)[1].lower()
	try:
		# Special key to force MediaPipe face detector
		if model_path.strip().lower() in {"mediapipe", "mp", "mediapipe-face"}:
			return MediaPipeFacePredictor()

		# Allow Ultralytics to auto-download if given a known model name
		if ext in {'.pt', '.pth'} and YOLO is not None:
			return YoloPredictor(model_path)

		# For formats that require a local file, ensure it exists and is non-empty
		if not os.path.isfile(model_path) or os.path.getsize(model_path) == 0:
			return DummyPredictor()

		if ext in {'.h5', '.keras'}:
			return KerasDensityPredictor(model_path)
		if ext == '.onnx':
			return OnnxPredictor(model_path)
	except Exception:
		# Any loading error falls back to auto
		return load_predictor_auto()

	# Unknown extension => auto
	return load_predictor_auto()
