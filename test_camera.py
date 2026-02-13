import cv2

# Hardcoded URLs from tracker.py
TOP_CAM_URL = "http://10.185.236.38:8080/video"
SIDE_CAM_URL = "http://10.185.236.109:8080/video"

def test_cameras():
    print(f"Connecting to Top Camera: {TOP_CAM_URL}...")
    cap1 = cv2.VideoCapture(TOP_CAM_URL)
    
    print(f"Connecting to Side Camera: {SIDE_CAM_URL}...")
    cap2 = cv2.VideoCapture(SIDE_CAM_URL)
    
    if not cap1.isOpened():
        print("FAILED: Could not open Top Camera")
    else:
        print("SUCCESS: Top Camera connected")
        
    if not cap2.isOpened():
        print("FAILED: Could not open Side Camera")
    else:
        print("SUCCESS: Side Camera connected")
        
    print("Press 'q' to exit")
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if ret1:
            frame1 = cv2.resize(frame1, (480, 360))
            cv2.imshow("Top Camera", frame1)
            
        if ret2:
            frame2 = cv2.resize(frame2, (480, 360))
            cv2.imshow("Side Camera", frame2)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_cameras()
