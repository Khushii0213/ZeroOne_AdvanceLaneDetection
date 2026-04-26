# Advanced Lane Detection System
### Classical Computer Vision — Extended Project

> Base repo: [naokishibuya/car-finding-lane-lines](https://github.com/naokishibuya/car-finding-lane-lines)  
> Extended to an advanced, robust, adverse-weather-aware system

---

## 1. Need Analysis

### 1.1 Problem Statement
Lane detection is a safety-critical component of Advanced Driver Assistance Systems (ADAS). The base repository demonstrates the core idea well, but it fails in several real-world scenarios:

| Failure mode | Why it happens in the base repo |
|---|---|
| Curved roads | Uses straight-line Hough extrapolation; curves are misfit |
| Night driving | Fixed HSL thresholds miss dim markings |
| Fog / haze | Contrast drops below Canny sensitivity |
| Rain / puddles | Specular reflections create false edges |
| Lane departure | No offset measurement or warning system |
| Flickering output | No temporal smoothing between frames |

### 1.2 Target Users
- Embedded ADAS engineers prototyping on dashcam footage  
- Computer Vision students demonstrating a complete classical pipeline  
- Researchers studying adverse-weather degradation on lane detection

### 1.3 Scope
This project extends the base into a **fully classical CV pipeline** (no deep learning), deliberately bounded to techniques covered in the course syllabus, while solving the gaps listed above.

---

## 2. System Architecture

```
Frame Input
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  1. SceneClassifier          (HSL histogram statistics)      │
│     → weather_mode: normal | foggy | rainy | night           │
│     → road_wetness: dry | wet | icy                          │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  2. ImageEnhancer            (adaptive preprocessing)        │
│     normal → passthrough                                     │
│     foggy  → Dark Channel Prior dehazing                     │
│     night  → gamma correction + CLAHE                        │
│     rainy  → bilateral filter + CLAHE                        │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  3. ColorThresholder         (multi-space masking)           │
│     HSL white mask + HSL/LAB yellow mask                     │
│     + adaptive threshold fallback (night/rain)               │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  4. EdgeDetector             (Canny + Sobel magnitude)       │
│     Canny for strong edges                                    │
│     Sobel magnitude for low-contrast edges (fog/night)       │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  5. ROISelector              (vanishing-point driven)        │
│     Hough lines in upper half → median intersection          │
│     Dynamic trapezoid (not a hard-coded polygon)             │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  6. LaneFitter               (polynomial regression)        │
│     Hough seeding → pixel segregation by slope/position      │
│     np.polyfit degree-2 → handles curves                     │
│     Exponential temporal smoothing over 15-frame history     │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  7. DepartureWarner          (real-world metric)             │
│     Lane centre offset in metres (XM/PX scale factor)        │
│     Warning threshold: ±0.4 m                                │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  8. Visualiser               (annotated output)              │
│     Filled lane polygon + left/right boundary lines          │
│     HUD panel: offset, curvature, scene, confidence          │
│     Red overlay + banner on departure warning                │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Mapping to Classical CV Syllabus

| Syllabus topic | Where used in this project |
|---|---|
| **Fourier Transform / Filtering** | Gaussian blur kernel in EdgeDetector; CLAHE via contrast-limiting histogram equalization |
| **Image Enhancement / Restoration** | CLAHE for night/rain; Dark Channel Prior for fog; gamma correction |
| **Histogram Processing** | SceneClassifier uses HSL histogram statistics; CLAHE internally uses histograms |
| **Edges — Canny, LOG, DOG** | EdgeDetector: Canny with weather-adaptive thresholds |
| **Orientation Histogram** | Implicit in HoughLinesP accumulator (votes by ρ, θ) |
| **Hough Transform** | LaneFitter seeds pixel sets via HoughLinesP; ROISelector finds vanishing point via Hough intersections |
| **Corners — Harris / Affine** | Vanishing point estimation uses line–line intersections (projective geometry) |
| **Image Segmentation / Region Growing** | ROI trapezoid = region-of-interest segmentation; colour masking = region growing by colour proximity |
| **MRFs / Edge-based Segmentation** | Sobel gradient magnitude used as edge-based segment boundary |
| **Pattern Analysis — Classification** | Slope-based classification of Hough lines into left/right lanes |
| **Dimensionality Reduction (PCA-like)** | Polynomial fitting reduces many edge pixels to 3 coefficients |
| **Motion Analysis — Temporal** | Exponentially weighted moving average over 15 frames emulates optical-flow-style temporal coherence |
| **Projective / Affine Transform** | Vanishing-point computation relies on projective line intersection |
| **Color Spaces (HSL, LAB, RGB)** | SceneClassifier + ColorThresholder use HSL and LAB spaces |

---

## 4. Adverse Weather Handling — Design Decisions

### 4.1 Fog
**Problem**: Fog scatters light uniformly, collapsing contrast. Canny edges become noise.  
**Solution**: Dark Channel Prior dehazing (He et al., 2009) — a classical restoration technique that estimates the scene's transmission map and inverts the haze model. No deep learning needed.  
**Limitation**: Fails on very thick fog (transmission < 0.1); requires at least partial scene texture.

### 4.2 Night
**Problem**: Low luminance → most colour thresholds return empty masks.  
**Solution**: Gamma correction (γ = 2.0) amplifies mid-tone road markings; CLAHE then locally equalises contrast without over-brightening headlight glare.  
**Limitation**: Oncoming headlights create bright blobs that can be misclassified as white markings.

### 4.3 Rain
**Problem**: Wet road creates specular mirror reflections. Raindrops add fine texture noise.  
**Solution**: Bilateral filter removes rain-drop noise while preserving lane edges (edge-preserving smoothing); CLAHE handles contrast collapse; wetness proxy triggers a warning to the driver HUD.  
**Limitation**: Heavy rain curtains with low visibility behave like fog; the pipeline degrades gracefully by relying more on temporal history.

### 4.4 Temporal Fallback
In any weather, if a frame produces fewer than 4 candidate points for one side, the system falls back to the previous frame's polynomial with a decaying confidence score. After several consecutive misses, confidence drops below 50 % and the HUD reflects this.

---

## 5. Drawbacks & Mitigations

| Drawback | Mitigation built-in | Future fix |
|---|---|---|
| Only straight/gentle curves detected | 2nd-order polynomial handles moderate curves | 3rd-order or B-spline for sharp turns |
| Fixed camera mount assumed | Vanishing-point ROI adapts to road grade changes | Homography-based bird's-eye view (IPM) |
| No lane count (multi-lane) | Architecture supports N lines; fitter currently does L+R | Cluster Hough lines by angle-groups |
| Occlusion by vehicles | Temporal history bridges short gaps | Kalman filter with constant-curvature motion model |
| Snow (white-on-white) | Not handled | Thermal IR channel or texture gradient |
| No calibration data | Scale factor is approximate | Camera calibration with chessboard |

---

## 6. File Structure

```
advanced_lane_detection/
├── src/
│   ├── __init__.py
│   └── lane_detector.py        ← all 8 pipeline stages
├── tests/
│   └── test_pipeline.py        ← 24 unit tests (all pass)
├── output/                     ← generated annotated images/videos
├── demo.py                     ← CLI runner
├── requirements.txt
└── README.md
```

---

## 7. Quick Start

```bash
pip install -r requirements.txt

# Run on synthetic frames (no video needed)
python demo.py --mode synth

# Run on your own image
python demo.py --mode image --input road.jpg

# Run on a video file
python demo.py --mode video --input highway.mp4

# Run unit tests
python -m pytest tests/ -v
```

---

## 8. Key Extensions over Base Repo

| Feature | Base repo | This project |
|---|---|---|
| Lane shape | Straight lines only | 2nd-order polynomial (curves) |
| Color masking | RGB white/yellow | HSL + LAB + adaptive threshold |
| Weather handling | None | 4 modes with dedicated preprocessing |
| ROI | Hard-coded trapezoid | Vanishing-point driven, dynamic |
| Temporal stability | None | 15-frame exponential smoothing |
| Lane departure | None | Metric offset + warning system |
| Road wetness | None | Luminance variance heuristic |
| Curvature | None | Radius of curvature in metres |
| Confidence | None | Per-frame score shown in HUD |
| Tests | None | 24 pytest unit tests |

---

## 9. References

1. He, K., Sun, J., Tang, X. (2009). *Single Image Haze Removal Using Dark Channel Prior.* CVPR.  
2. Canny, J. (1986). *A Computational Approach to Edge Detection.* IEEE TPAMI.  
3. Duda, R.O., Hart, P.E. (1972). *Use of the Hough Transformation to Detect Lines.* CACM.  
4. Shi, J. (1994). *Good Features to Track.* CVPR. (Harris variant)  
5. OpenCV documentation — `HoughLinesP`, `CLAHE`, `bilateralFilter`.
