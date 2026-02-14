import cv2
import numpy as np

CAM_SRC = 0  # Default to 0 (internal webcam) for TOP view
SIDE_CAM_SRC = 1 # Default to 1 (second webcam) for SIDE view

# You can hardcode your camera URLs here if you don't want to type them in the website
DEFAULT_TOP_CAM_URL =  "http://172.18.232.196:8080/video"
DEFAULT_SIDE_CAM_URL = "http://172.18.230.176:8080/video"


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

    if not cap.isOpened():
        print(f"ERROR: Failed to open Top Camera ({top_source})")
        return {
            "psi": 0, "tremor": 0, "error": 0, "depth_error": 0,
            "pressure_dev": 0, "over_pen": 0, "skill": "Camera Error", 
            "mode": mode, "trajectory": []
        }
    else:
        print("SUCCESS: Top Camera connected.")

    # Determine Side Camera Source logic with fallback
    side_source_primary = None
    if side_cam_url and len(side_cam_url) > 5:
        side_source_primary = side_cam_url
    elif DEFAULT_SIDE_CAM_URL and len(DEFAULT_SIDE_CAM_URL) > 5:
        side_source_primary = DEFAULT_SIDE_CAM_URL

    cap_side = None
    
    # Try primary side source (IP)
    if side_source_primary:
        print(f"Connecting to Side Camera (Primary): {side_source_primary}")
        temp_cap = cv2.VideoCapture(side_source_primary)
        temp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if temp_cap.isOpened():
            cap_side = temp_cap
            print("SUCCESS: Side Camera connected (Primary).")
        else:
            print(f"WARNING: Failed to connect to Side Camera (Primary): {side_source_primary}")
            temp_cap.release()
    
    # Fallback to local index if primary failed or wasn't set
    if cap_side is None:
        print(f"Connecting to Side Camera (Fallback): Index {SIDE_CAM_SRC}")
        temp_cap = cv2.VideoCapture(SIDE_CAM_SRC)
        if temp_cap.isOpened():
            cap_side = temp_cap
            print(f"SUCCESS: Side Camera connected (Fallback Index {SIDE_CAM_SRC}).")
        else:
            print(f"WARNING: Failed to connect to Side Camera (Fallback Index {SIDE_CAM_SRC}). Side view disabled.")
            temp_cap.release()
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
        # COLOR TRACKING (MOVED UP)
        # =======================
        x, y, radius, area = detect_blue_object(frame)

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
            # Target Path
            cv2.line(frame, (50, 240), (w - 50, 240), (0, 255, 0), 2)
            
            # Start/End Markers
            cv2.circle(frame, (50, 240), 10, (0, 255, 0), -1) # Green Start
            cv2.putText(frame, "START", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.circle(frame, (w - 50, 240), 10, (0, 0, 255), -1) # Red End
            cv2.putText(frame, "END", (w - 70, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        elif mode == "circle":
            cv2.circle(frame, (w // 2, h // 2), 100, (0, 255, 0), 2)
            
            # Start at 3 o'clock position
            sx, sy = w // 2 + 100, h // 2
            cv2.circle(frame, (sx, sy), 10, (0, 255, 0), -1)
            cv2.putText(frame, "START/END", (sx + 15, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        elif mode == "micro":
            cv2.circle(frame, (w // 2, h // 2), 40, (0, 255, 0), 2)
            # Center is target
            cv2.putText(frame, "TARGET", (w // 2 - 30, h // 2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        elif mode == "brain":
            cv2.polylines(frame, [brain_path], False, (0, 255, 0), 2)

            # Start/End from path
            bsx, bsy = brain_path[0]
            bex, bey = brain_path[-1]
            
            cv2.circle(frame, (bsx, bsy), 8, (0, 255, 0), -1)
            cv2.putText(frame, "START", (bsx - 20, bsy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.circle(frame, (bex, bey), 8, (0, 0, 255), -1)
            cv2.putText(frame, "END", (bex + 10, bey), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Restricted zone
            rx1, ry1 = 300, 130
            rx2, ry2 = 420, 240
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
            cv2.putText(frame, "Restricted Zone", (rx1, ry1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        elif mode == "angle":
            cv2.rectangle(frame, (w // 2 - 60, 200), (w // 2 + 60, 300), (255, 0, 0), 2)

        elif mode == "suturing":
            # Draw lines
            cv2.line(frame, (100, 220), (540, 220), (0, 255, 0), 2)
            cv2.line(frame, (100, 280), (540, 280), (0, 255, 0), 2)
            
            # Start/End for top line
            cv2.circle(frame, (100, 220), 8, (0, 255, 0), -1)
            cv2.putText(frame, "START", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.circle(frame, (540, 220), 8, (0, 0, 255), -1)
            cv2.putText(frame, "END", (530, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        elif mode == "depth_drill":
            cv2.putText(frame, "Maintain Depth Zone",
                        (180, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        elif mode == "needle_target":
            cv2.circle(frame, (w // 2, h // 2), 12, (0, 255, 0), -1)
            cv2.putText(frame, "TARGET", (w // 2 - 30, h // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if x is not None and radius > 5:

            # =====================
            # REAL-TIME METRICS
            # =====================
            
            # Tremor (Instantaneous)
            instant_tremor = 0
            if prev is not None:
                instant_tremor = np.linalg.norm(np.array([x, y]) - np.array(prev))
                tremor.append(instant_tremor)
            prev = (x, y)

            # Error (Instantaneous)
            instant_error = 0
            if mode == "line":
                instant_error = abs(y - 240)
            elif mode == "circle":
                instant_error = abs(np.sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2) - 100)
            elif mode == "brain":
                 dist = cv2.pointPolygonTest(brain_path, (x, y), True)
                 instant_error = max(0, abs(dist) - 15) # 15 is tolerance
            elif mode == "needle_target":
                instant_error = np.sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2)
            
            errors.append(instant_error)

            # Depth
            if z_val is not None:
                depth_values.append(z_val)
            else:
                depth_values.append(radius)

            # Pressure
            pressure_values.append(area / 500.0)

            # =====================
            # DYNAMIC FEEDBACK
            # =====================
            
            # Color logic: Green (Good), Red (Bad)
            # Thresholds: Tremor > 15, Error > 20
            color = (0, 255, 0) # Green
            warning_text = ""
            
            if instant_tremor > 15:
                color = (0, 0, 255) # Red
                warning_text = "HIGH TREMOR!"
            elif instant_error > 20:
                color = (0, 0, 255) # Red
                warning_text = "OFF PATH!"
            elif mode == "brain" and (300 < x < 420) and (130 < y < 240):
                color = (0, 0, 255)
                warning_text = "RESTRICTED AREA!"

            # Store (x, y, z, color)
            # Use z_val if available, else use radius as depth proxy, else 0
            current_z = z_val if z_val is not None else radius
            trajectory.append((x, y, current_z, color))

            # Draw Trajectory with dynamic colors
            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    # Use stored color for each segment
                    pt1 = trajectory[i-1][:2]
                    pt2 = trajectory[i][:2]
                    seg_color = trajectory[i][3] # Color is now at index 3
                    cv2.line(frame, pt1, pt2, seg_color, 2)
            
            # Draw Warning
            if warning_text:
                cv2.putText(frame, warning_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 3)

            # =====================
            # PROGRESS LOGIC
            # =====================

            if mode == "line":
                if instant_error < 12:
                    progress_counter += 1

            elif mode == "circle":
                if instant_error < 12:
                    progress_counter += 1

            elif mode == "brain":
                # Restricted penalty
                if (300 < x < 420) and (130 < y < 240):
                    restricted_hits += 1
                
                dist = cv2.pointPolygonTest(brain_path, (x, y), True)
                if abs(dist) < 15:
                    progress_counter += 1

            elif mode == "suturing":
                stitch_spacing = abs((x % 60) - 30)
                stitch_scores.append(stitch_spacing)
                if stitch_spacing < 10:
                    progress_counter += 1

            elif mode == "needle_target":
                targeting_errors.append(instant_error)
                if instant_error < 12:
                    progress_counter += 1

            elif mode == "depth_drill":
                if 15 < radius < 40:
                    progress_counter += 1

            cv2.circle(frame, (x, y), 6, color, -1)

        # Draw HUD / Progress Bar
        bar_width = int((progress_counter / required_progress) * w)
        cv2.rectangle(frame, (0, h - 20), (bar_width, h), (0, 255, 0), -1)
        cv2.putText(frame, f"Progress: {int((progress_counter/required_progress)*100)}%", 
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


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