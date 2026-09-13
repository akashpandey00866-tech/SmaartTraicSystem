import cv2
import numpy as np
from ultralytics import YOLO

def run_pro_detector():
    print("="*60)
    print(" PRO-LEVEL AI TRAFFIC, HUMAN/ANIMAL & SENSOR MONITORING SYSTEM ")
    print("="*60)
    print("Select Input Source:")
    print("1. Laptop Webcam (Live)")
    print("2. Traffic Video File (.mp4)")
    print("3. Mobile Camera (via IP Webcam App URL)")
    
    choice = input("Enter your choice (1, 2, or 3): ").strip()
    
    source = 0
    if choice == "1":
        source = 0
        print("[INFO] Opening Laptop Webcam...")
    elif choice == "2":
        source = input("Enter video file name (e.g., traffic.mp4): ").strip()
        print(f"[INFO] Opening video file: {source}")
    elif choice == "3":
        source = input("Enter Mobile IP Webcam URL (e.g., http://192.168.1.5:8080/video): ").strip()
        print(f"[INFO] Connecting to Mobile Camera at {source}...")
    else:
        print("[WARNING] Invalid choice! Defaulting to Laptop Webcam.")
        source = 0

    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Could not open the selected camera/video source!")
        return

    print("\n[SUCCESS] Pro System Running! Press 'q' on the video window to exit.\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("[INFO] Stream ended or feed interrupted.")
            break

        # Resize frame for smooth processing
        frame = cv2.resize(frame, (1020, 600))
        
        # Run YOLO inference
        results = model(frame)
        boxes = results[0].boxes
        
        # Counters
        car_count = 0
        person_count = 0
        dog_count = 0
        bike_count = 0
        vehicle_boxes = []

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            if conf < 0.4:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # Categorize objects with dynamic custom colors
            if class_name in ['car', 'truck', 'bus']:
                car_count += 1
                vehicle_boxes.append((x1, y1, x2, y2))
                color = (0, 255, 0) # Green for Vehicles
            elif class_name == 'person':
                person_count += 1
                color = (255, 0, 0) # Blue for Humans
            elif class_name in ['dog', 'cat']:
                dog_count += 1
                color = (0, 255, 255) # Yellow for Animals/Dogs
            elif class_name in ['motorcycle', 'bicycle']:
                bike_count += 1
                vehicle_boxes.append((x1, y1, x2, y2))
                color = (0, 165, 255) # Orange for Bikes
            else:
                color = (180, 180, 180)

            # Draw bounding box & exact dynamic name
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name.upper()} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 🚨 COLLISION DETECTION LOGIC 🚨
        collision_detected = False
        for i in range(len(vehicle_boxes)):
            for j in range(i + 1, len(vehicle_boxes)):
                boxA, boxB = vehicle_boxes[i], vehicle_boxes[j]
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])
                
                interArea = max(0, xB - xA) * max(0, yB - yA)
                if interArea > 400: # Overlap threshold for accident
                    collision_detected = True
                    cv2.rectangle(frame, (xA-5, yA-5), (xB+5, yB+5), (0, 0, 255), 3)

        # ================= SENSOR SIMULATION DATA =================
        # Simulating virtual IoT sensors based on real-time vehicle density
        total_density = car_count + bike_count
        simulated_speed = max(5, 60 - (total_density * 4)) # Speed drops as density rises
        simulated_co2 = round(total_density * 0.035, 2)    # Carbon emissions scale with traffic
        proximity_status = "SAFE" if not collision_detected else "CRITICAL ACCIDENT"

        # Display Top Dashboard Stats Overlay
        cv2.putText(frame, f"Cars: {car_count} | Bikes: {bike_count} | Humans: {person_count} | Dogs: {dog_count}", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display Sensor Data Panel at Bottom
        sensor_text = f"[SENSORS] Speed: {simulated_speed} km/h | CO2: {simulated_co2} kg/m | Status: {proximity_status}"
        cv2.putText(frame, sensor_text, (20, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Flash Collision / Accident Warning Alert
        if collision_detected:
            cv2.putText(frame, "!!! ACCIDENT / PROXIMITY COLLISION DETECTED !!!", (120, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        # Show Window
        cv2.imshow("Pro-Level AI Traffic & Sensor Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pro_detector()