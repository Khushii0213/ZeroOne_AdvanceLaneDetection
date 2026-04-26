import cv2
import numpy as np
import time

prev_angle = 0
prev_left_fit = None
prev_right_fit = None
departure_counter = 0

# 🔥 LOAD HAAR CASCADE
car_cascade = cv2.CascadeClassifier('cars.xml')


# ==============================
# THRESHOLDING
# ==============================
def thresholding(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    white = cv2.inRange(hls, (0,200,0), (255,255,255))  
    yellow = cv2.inRange(hls, (15,30,115), (35,204,255))

    return cv2.bitwise_or(edges, cv2.bitwise_or(white, yellow))


# ==============================
# WARP
# ==============================
def warp(img):
    h, w = img.shape[:2]

    src = np.float32([
        [w*0.43, h*0.65],
        [w*0.57, h*0.65],
        [w*0.9, h],
        [w*0.1, h]
    ])

    dst = np.float32([[0,0],[w,0],[w,h],[0,h]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, (w,h))

    return warped, Minv


# ==============================
# SLIDING WINDOW + SMOOTHING
# ==============================
def sliding_window(binary):
    global prev_left_fit, prev_right_fit

    histogram = np.sum(binary[binary.shape[0]//2:, :], axis=0)
    midpoint = histogram.shape[0]//2

    nonzero = binary.nonzero()
    y = np.array(nonzero[0])
    x = np.array(nonzero[1])

    left_inds = x < midpoint
    right_inds = x >= midpoint

    if len(x[left_inds]) > 50:
        left_fit = np.polyfit(y[left_inds], x[left_inds], 2)
    else:
        left_fit = prev_left_fit

    if len(x[right_inds]) > 50:
        right_fit = np.polyfit(y[right_inds], x[right_inds], 2)
    else:
        right_fit = prev_right_fit

    if prev_left_fit is not None:
        left_fit = 0.7*prev_left_fit + 0.3*left_fit
    if prev_right_fit is not None:
        right_fit = 0.7*prev_right_fit + 0.3*right_fit

    prev_left_fit = left_fit
    prev_right_fit = right_fit

    return left_fit, right_fit


# ==============================
# METRICS
# ==============================
def compute_metrics(left_fit, right_fit, width):
    global prev_angle

    y = 720

    left_x = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
    right_x = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]

    lane_center = (left_x + right_x)/2
    frame_center = width/2

    offset_px = frame_center - lane_center

    xm_per_pix = 3.7 / 700
    offset_m = offset_px * xm_per_pix

    angle = np.arctan(offset_px/y)*180/np.pi
    angle = 0.7*prev_angle + 0.3*angle
    prev_angle = angle

    lane_width = abs(right_x - left_x)

    return angle, offset_m, lane_width


# ==============================
# DRAW LANE + CENTER PATH
# ==============================
def draw_lane(img, binary, left_fit, right_fit, Minv):
    h, w = binary.shape

    ploty = np.linspace(0, h-1, h)

    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    lane = np.zeros((h,w,3), dtype=np.uint8)

    pts = np.hstack((
        np.array([np.transpose(np.vstack([leftx, ploty]))]),
        np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])))

    cv2.fillPoly(lane, np.int32([pts]), (0,255,100))

    center = (leftx + rightx)/2
    for i in range(0, len(ploty), 30):
        cv2.arrowedLine(lane,
                        (int(center[i]), int(ploty[i])),
                        (int(center[i]), int(ploty[i]-15)),
                        (255,255,0), 2)

    newwarp = cv2.warpPerspective(lane, Minv, (w,h))

    return cv2.addWeighted(img, 1, newwarp, 0.4, 0)


# ==============================
# OBJECT DETECTION (FILTERED)
# ==============================
def detect_objects(frame):
    detections = []

    h, w = frame.shape[:2]

    # 🔥 ROI: only bottom half
    roi = frame[int(h*0.5):h, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)

    for (x, y, w_box, h_box) in cars:

        # 🔥 SIZE FILTER
        if w_box < 40 or h_box < 40:
            continue

        # 🔥 ASPECT RATIO FILTER
        aspect_ratio = w_box / h_box
        if aspect_ratio < 1.0 or aspect_ratio > 3.5:
            continue

        # adjust y to original frame
        y = y + int(h*0.5)

        distance = 1000 / (h_box + 1)

        detections.append((x, y, x+w_box, y+h_box, distance))

    return detections


def draw_objects(frame, detections):
    for (x1, y1, x2, y2, dist) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{int(dist)} m",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)
    return frame


# ==============================
# CONFIDENCE
# ==============================
def compute_confidence(lane_width):
    return max(0, min(100, 100 - abs(lane_width-700)/5))


# ==============================
# MAIN PIPELINE
# ==============================
def process_frame(frame, fps):
    global departure_counter

    thresh = thresholding(frame)
    warped, Minv = warp(thresh)

    left_fit, right_fit = sliding_window(warped)

    result = draw_lane(frame, warped, left_fit, right_fit, Minv)

    angle, offset, lane_width = compute_metrics(left_fit, right_fit, frame.shape[1])
    confidence = compute_confidence(lane_width)

    if abs(offset) > 0.5:
        departure_counter += 1
    else:
        departure_counter = max(0, departure_counter-1)

    if departure_counter > 5:
        state = "DANGER"
    elif departure_counter > 2:
        state = "WARNING"
    else:
        state = "SAFE"

    if state != "SAFE":
        overlay = result.copy()
        cv2.rectangle(overlay, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), -1)
        alpha = 0.2 if state == "WARNING" else 0.35
        result = cv2.addWeighted(overlay, alpha, result, 1-alpha, 0)

        cv2.putText(result, f"⚠ {state} LANE DEPARTURE!", (50,200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    if angle < -5:
        direction = "LEFT"
    elif angle > 5:
        direction = "RIGHT"
    else:
        direction = "STRAIGHT"

    cv2.putText(result, f"Angle: {int(angle)} deg", (30,40), 0, 1, (0,255,255), 2)
    cv2.putText(result, f"Offset: {offset:.2f} m", (30,80), 0, 1, (255,255,0), 2)
    cv2.putText(result, f"Direction: {direction}", (30,120), 0, 1, (255,0,255), 2)
    cv2.putText(result, f"Confidence: {int(confidence)}%", (30,160), 0, 1, (0,255,100), 2)
    cv2.putText(result, f"State: {state}", (30,200), 0, 1, (0,0,255), 2)
    cv2.putText(result, f"FPS: {int(fps)}", (30,240), 0, 1, (0,255,0), 2)

    # 🔥 OBJECT DETECTION FINAL
    detections = detect_objects(result)
    result = draw_objects(result, detections)

    return result


# ==============================
# RUN
# ==============================
cap = cv2.VideoCapture("input/video.mp4")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time

    output = process_frame(frame, fps)

    cv2.imshow("ADAS Lane Detection (No DL)", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
