import cv2
import numpy as np

CAM_SRC = 0  # Default to 0 (internal webcam) if no URL provided

# You can hardcode your camera URLs here if you don't want to type them in the website
DEFAULT_TOP_CAM_URL =  "http://10.185.236.38:8080/video"
DEFAULT_SIDE_CAM_URL = "http://10.185.236.109:8080/video"


def classify_skill(psi):
    if psi >= 80:
        return "Expert"
    elif psi >= 60:
        return "Intermediate"
    else:
        return "Beginner"


def detect_blue_object(frame):
    if frame is None:
        return None, None, None, None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 120, 70])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        area = cv2.contourArea(c)
        return int(x), int(y), radius, area
    
    return None, None, None, None


def start_tracking(mode, side_cam_url=None, top_cam_url=None):

    # Determine Top Camera Source
    top_source = CAM_SRC
    if top_cam_url and len(top_cam_url) > 5:
        top_source = top_cam_url
    elif DEFAULT_TOP_CAM_URL and len(DEFAULT_TOP_CAM_URL) > 5:
        top_source = DEFAULT_TOP_CAM_URL
        
    cap = cv2.VideoCapture(top_source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Determine Side Camera Source
    side_source = None
    if side_cam_url and len(side_cam_url) > 5:
        side_source = side_cam_url
    elif DEFAULT_SIDE_CAM_URL and len(DEFAULT_SIDE_CAM_URL) > 5:
        side_source = DEFAULT_SIDE_CAM_URL

    cap_side = None
    if side_source:
        cap_side = cv2.VideoCapture(side_source)
        cap_side.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        return {
            "psi": 0,
            "tremor": 0,
            "error": 0,
            "depth_error": 0,
            "pressure_dev": 0,
            "over_pen": 0,
            "skill": "Camera Error",
            "mode": mode,
            "trajectory": []
        }

    tremor = []
    errors = []
    depth_values = []
    pressure_values = []
    trajectory = []
    prev = None

    restricted_hits = 0
    stitch_scores = []
    targeting_errors = []

    progress_counter = 0
    required_progress = 180

    # Brain path coordinates
    brain_path = np.array([[80, 300], [200, 180], [350, 260], [520, 160]])

    while True:

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # =======================
        # SIDE CAMERA PROCESSING
        # =======================
        z_val = None
        if cap_side:
            ret_side, frame_side = cap_side.read()
            if ret_side:
                frame_side = cv2.resize(frame_side, (640, 480))
                # Perform detection on side view
                sx, sy, sr, sa = detect_blue_object(frame_side)
                
                if sx is not None:
                    # Use X-coordinate of side view as Z-depth
                    z_val = sx
                    
                    cv2.circle(frame_side, (sx, sy), 5, (0, 255, 255), -1)
                    cv2.putText(frame_side, f"Z: {sx}", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Side View (Z-Axis)", frame_side)

        # =======================
        # MODE VISUAL OVERLAYS
        # =======================

        if mode == "line":
            cv2.line(frame, (50, 240), (w - 50, 240), (0, 255, 0), 2)

        elif mode == "circle":
            cv2.circle(frame, (w // 2, h // 2), 100, (0, 255, 0), 2)

        elif mode == "micro":
            cv2.circle(frame, (w // 2, h // 2), 40, (0, 255, 0), 2)

        elif mode == "brain":
            cv2.polylines(frame, [brain_path], False, (0, 255, 0), 2)

            # Restricted zone
            rx1, ry1 = 300, 130
            rx2, ry2 = 420, 240
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
            cv2.putText(frame, "Restricted Zone", (rx1, ry1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        elif mode == "angle":
            cv2.rectangle(frame, (w // 2 - 60, 200), (w // 2 + 60, 300), (255, 0, 0), 2)

        elif mode == "suturing":
            cv2.line(frame, (100, 220), (540, 220), (0, 255, 0), 2)
            cv2.line(frame, (100, 280), (540, 280), (0, 255, 0), 2)

        elif mode == "depth_drill":
            cv2.putText(frame, "Maintain Depth Zone",
                        (180, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        elif mode == "needle_target":
            cv2.circle(frame, (w // 2, h // 2), 12, (0, 255, 0), -1)

        # =======================
        # COLOR TRACKING
        # =======================

        x, y, radius, area = detect_blue_object(frame)

        if x is not None and radius > 5:

            trajectory.append((x, y))

            if len(trajectory) > 1:
                cv2.line(frame, trajectory[-2], trajectory[-1],
                            (0, 255, 255), 2)

            # Tremor
            if prev is not None:
                tremor.append(np.linalg.norm(
                    np.array([x, y]) - np.array(prev)))
            prev = (x, y)

            # Error (baseline deviation)
            errors.append(abs(y - 240))

            # Depth
            if z_val is not None:
                depth_values.append(z_val)
            else:
                depth_values.append(radius)

            # Pressure
            pressure_values.append(area / 500.0)

            # =====================
            # MODE SPECIFIC LOGIC
            # =====================

            if mode == "line":
                if abs(y - 240) < 12:
                    progress_counter += 1

            elif mode == "circle":
                dist = abs(np.sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2) - 100)
                if dist < 12:
                    progress_counter += 1

            elif mode == "brain":

                # Restricted penalty
                if (300 < x < 420) and (130 < y < 240):
                    restricted_hits += 1
                    cv2.putText(frame, "WARNING: Restricted Area!",
                                (160, 70),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)

                dist = cv2.pointPolygonTest(brain_path, (x, y), True)
                if abs(dist) < 15:
                    progress_counter += 1

            elif mode == "suturing":
                stitch_spacing = abs((x % 60) - 30)
                stitch_scores.append(stitch_spacing)
                if stitch_spacing < 10:
                    progress_counter += 1

            elif mode == "needle_target":
                target_error = np.sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2)
                targeting_errors.append(target_error)
                if target_error < 12:
                    progress_counter += 1

            elif mode == "depth_drill":
                if 15 < radius < 40:
                    progress_counter += 1

            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)

        cv2.imshow("AI Surgical Trainer", frame)

        if progress_counter >= required_progress:
            break

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    if cap_side:
        cap_side.release()
    cv2.destroyAllWindows()

    # =====================
    # METRIC CALCULATION
    # =====================

    tremor_score = np.std(tremor) if len(tremor) > 5 else 0
    error_score = np.mean(errors) if len(errors) > 5 else 0
    depth_error = np.std(depth_values) if len(depth_values) > 5 else 0
    pressure_dev = np.std(pressure_values) if len(pressure_values) > 5 else 0
    over_pen = sum(p > 1.3 for p in pressure_values)

    stitch_accuracy = 100 - np.mean(stitch_scores) if stitch_scores else None
    targeting_accuracy = 100 - np.mean(targeting_errors) if targeting_errors else None
    depth_variation_index = depth_error if mode == "depth_drill" else None

    # PSI Calculation
    psi = 100 - (
        tremor_score * 0.35 +
        error_score * 0.35 +
        depth_error * 0.15 +
        pressure_dev * 5 +
        restricted_hits * 0.2
    )

    psi = max(0, min(100, psi))

    result = {
        "psi": round(psi, 2),
        "tremor": round(tremor_score, 2),
        "error": round(error_score, 2),
        "depth_error": round(depth_error, 2),
        "pressure_dev": round(pressure_dev, 2),
        "over_pen": over_pen,
        "skill": classify_skill(psi),
        "mode": mode,
        "trajectory": trajectory
    }

    if stitch_accuracy is not None:
        result["stitch_accuracy"] = round(stitch_accuracy, 2)

    if targeting_accuracy is not None:
        result["targeting_accuracy"] = round(targeting_accuracy, 2)

    if depth_variation_index is not None:
        result["depth_variation_index"] = round(depth_variation_index, 2)

    return result
