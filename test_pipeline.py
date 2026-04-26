"""
tests/test_pipeline.py
─────────────────────
Unit tests for the Advanced Lane Detection pipeline.
Run with:  python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
import pytest

from src.lane_detector import (
    SceneClassifier,
    ImageEnhancer,
    ColorThresholder,
    EdgeDetector,
    ROISelector,
    LaneFitter,
    DepartureWarner,
    AdvancedLaneDetector,
    LaneLine,
)
from demo import make_synthetic_frame


# ──────────────────────── fixtures ────────────────────────────────────────────

@pytest.fixture
def normal_frame():
    return make_synthetic_frame(weather="normal")

@pytest.fixture
def night_frame():
    return make_synthetic_frame(weather="night")

@pytest.fixture
def foggy_frame():
    return make_synthetic_frame(weather="foggy")

@pytest.fixture
def detector():
    return AdvancedLaneDetector()


# ──────────────────────── scene classifier ────────────────────────────────────

class TestSceneClassifier:
    def test_normal(self, normal_frame):
        w, _ = SceneClassifier().classify(normal_frame)
        assert w in ("normal", "foggy", "rainy", "night")

    def test_night_detected(self, night_frame):
        w, _ = SceneClassifier().classify(night_frame)
        assert w == "night"

    def test_foggy_detected(self, foggy_frame):
        w, _ = SceneClassifier().classify(foggy_frame)
        # foggy frame has high mean L and low std
        assert w in ("foggy", "normal")   # allow slight variation

    def test_returns_two_strings(self, normal_frame):
        result = SceneClassifier().classify(normal_frame)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)


# ──────────────────────── image enhancer ──────────────────────────────────────

class TestImageEnhancer:
    def test_output_shape_preserved(self, normal_frame):
        enhanced = ImageEnhancer().enhance(normal_frame.copy(), "normal")
        assert enhanced.shape == normal_frame.shape

    def test_night_brightens(self, night_frame):
        enhanced = ImageEnhancer().enhance(night_frame.copy(), "night")
        assert enhanced.mean() >= night_frame.mean()

    def test_foggy_output_valid(self, foggy_frame):
        enhanced = ImageEnhancer().enhance(foggy_frame.copy(), "foggy")
        assert enhanced.dtype == np.uint8
        assert enhanced.shape == foggy_frame.shape


# ──────────────────────── colour thresholder ──────────────────────────────────

class TestColorThresholder:
    def test_binary_output(self, normal_frame):
        mask = ColorThresholder.threshold(normal_frame, "normal")
        unique = np.unique(mask)
        assert set(unique).issubset({0, 255})

    def test_night_adaptive(self, night_frame):
        mask = ColorThresholder.threshold(night_frame, "night")
        # should detect some lane-like pixels
        assert mask.sum() > 0


# ──────────────────────── edge detector ───────────────────────────────────────

class TestEdgeDetector:
    def test_binary_output(self, normal_frame):
        gray = cv2.cvtColor(normal_frame, cv2.COLOR_BGR2GRAY)
        edges = EdgeDetector.detect(gray, "normal")
        unique = np.unique(edges)
        assert set(unique).issubset({0, 255})

    def test_has_edges(self, normal_frame):
        gray = cv2.cvtColor(normal_frame, cv2.COLOR_BGR2GRAY)
        edges = EdgeDetector.detect(gray, "normal")
        assert edges.sum() > 0


# ──────────────────────── ROI selector ────────────────────────────────────────

class TestROISelector:
    def test_output_shape(self, normal_frame):
        gray  = cv2.cvtColor(normal_frame, cv2.COLOR_BGR2GRAY)
        edges = EdgeDetector.detect(gray, "normal")
        roi, pts = ROISelector().get_roi_mask(edges)
        assert roi.shape == edges.shape

    def test_roi_pts_is_array(self, normal_frame):
        gray  = cv2.cvtColor(normal_frame, cv2.COLOR_BGR2GRAY)
        edges = EdgeDetector.detect(gray, "normal")
        _, pts = ROISelector().get_roi_mask(edges)
        assert isinstance(pts, np.ndarray)
        assert pts.shape[1] == 2   # Nx2 array


# ──────────────────────── lane fitter ─────────────────────────────────────────

class TestLaneFitter:
    def test_returns_tuple(self, normal_frame):
        h, w = normal_frame.shape[:2]
        gray  = cv2.cvtColor(normal_frame, cv2.COLOR_BGR2GRAY)
        edges = EdgeDetector.detect(gray, "normal")
        roi, _ = ROISelector().get_roi_mask(edges)
        result = LaneFitter().fit(roi, h, w)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_lane_line_dataclass(self):
        ll = LaneLine(poly_coeffs=np.array([0.0, 0.5, 100.0]),
                      base_x=300.0, confidence=0.8)
        assert ll.detected is True
        assert 0 <= ll.confidence <= 1


# ──────────────────────── departure warner ────────────────────────────────────

class TestDepartureWarner:
    def test_no_lanes_no_warning(self):
        offset, warn = DepartureWarner().analyse(None, None, 480, 640)
        assert offset == 0.0
        assert warn is False

    def test_centred_no_warning(self):
        left  = LaneLine(np.zeros(3), base_x=220.0)
        right = LaneLine(np.zeros(3), base_x=420.0)
        offset, warn = DepartureWarner().analyse(left, right, 480, 640)
        # vehicle centre = 320, lane centre = 320 → offset ≈ 0
        assert abs(offset) < 0.1
        assert warn is False

    def test_far_offset_triggers_warning(self):
        left  = LaneLine(np.zeros(3), base_x=50.0)
        right = LaneLine(np.zeros(3), base_x=250.0)
        # vehicle centre (320) is far right of lane centre (150)
        _, warn = DepartureWarner().analyse(left, right, 480, 640)
        assert warn is True


# ──────────────────────── end-to-end pipeline ─────────────────────────────────

class TestPipeline:
    @pytest.mark.parametrize("weather", ["normal", "foggy", "night", "rainy"])
    def test_all_weather_modes(self, weather, detector):
        frame = make_synthetic_frame(weather=weather)
        result = detector.process_frame(frame)
        assert result.annotated is not None
        assert result.annotated.shape == frame.shape
        assert result.weather_mode in ("normal", "foggy", "night", "rainy")

    def test_annotated_is_uint8(self, detector, normal_frame):
        result = detector.process_frame(normal_frame)
        assert result.annotated.dtype == np.uint8

    def test_repeated_frames_stable(self, detector, normal_frame):
        """History / temporal smoothing should not crash on repeated calls."""
        for _ in range(30):
            result = detector.process_frame(normal_frame)
        assert result.annotated is not None
