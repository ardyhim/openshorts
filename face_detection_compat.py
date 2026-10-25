"""Face detection that keeps working when MediaPipe cannot be installed.

The pipeline only needs the tiny legacy surface::

    detector = FaceDetection(model_selection=1, min_detection_confidence=0.5)
    result = detector.process(rgb_frame)
    for d in result.detections:
        box = d.location_data.relative_bounding_box

MediaPipe 0.10.21 is the last release with ``mp.solutions``. It has no wheel
for Python 3.13, which is what Colab now runs. On those runtimes this module
falls back to OpenCV's bundled Haar cascade: free, no model download, and it
returns the same relative-box structure. When a compatible MediaPipe is
installed (Python <= 3.12), it is used unchanged so existing deployments keep
BlazeFace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:  # MediaPipe is optional: Colab's Python 3.13 has no compatible wheel.
    import mediapipe as _mp
except Exception:  # pragma: no cover - depends on the runtime
    _mp = None


def _mediapipe_solutions():
    if _mp is None:
        return None
    return getattr(_mp, "solutions", None)


@dataclass
class _BoundingBox:
    xmin: float
    ymin: float
    width: float
    height: float


@dataclass
class _LocationData:
    relative_bounding_box: _BoundingBox


@dataclass
class _Detection:
    location_data: _LocationData


@dataclass
class _Result:
    detections: list = field(default_factory=list)


class FaceDetection:
    """Subset of ``mediapipe.solutions.face_detection.FaceDetection``."""

    backend = "none"

    def __init__(self, model_selection=1, min_detection_confidence=0.5):
        self._backend = None
        self._cascade = None

        solutions = _mediapipe_solutions()
        if solutions is not None and hasattr(solutions, "face_detection"):
            try:
                self._backend = solutions.face_detection.FaceDetection(
                    model_selection=model_selection,
                    min_detection_confidence=min_detection_confidence,
                )
                self.backend = "mediapipe"
                return
            except Exception:
                # Installed but unusable (missing native model, ABI mismatch).
                self._backend = None

        import cv2

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            raise RuntimeError(f"OpenCV face cascade not found: {cascade_path}")
        self._cascade = cascade
        self._min_detection_confidence = float(min_detection_confidence)
        self.backend = "opencv-haar"

    def process(self, rgb_frame):
        if self._backend is not None:
            return self._backend.process(rgb_frame)

        import cv2

        if rgb_frame is None or getattr(rgb_frame, "size", 0) == 0:
            return _Result([])

        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape[:2]
        if not height or not width:
            return _Result([])

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24),
        )
        detections = []
        for x, y, box_w, box_h in faces:
            detections.append(
                _Detection(
                    _LocationData(
                        _BoundingBox(
                            xmin=max(0.0, float(x) / width),
                            ymin=max(0.0, float(y) / height),
                            width=min(1.0, float(box_w) / width),
                            height=min(1.0, float(box_h) / height),
                        )
                    )
                )
            )
        return _Result(detections)
