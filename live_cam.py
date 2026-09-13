import cv2
from ultralytics import YOLO

def start_detection():
    print("--- AI TRAFFIC & VEHICLE DETECTION SYSTEM ---")
    print("Select Input Source:")
    print("1. Laptop Built-in Webcam")
    print("2. Local Video File (e.g., traffic.mp4)")
    print("3. Phone Camera (via IP Webcam app / URL)")
    
    choice = input("Enter your choice (1, 2, or 3): ").strip()
    
    source = 0 # Default laptop webcam
    
    if choice == "1":
        source = 0
        print("Opening Laptop Webcam...")
    elif choice == "2":
        file_path = input("Enter video file name (e.g., traffic.mp4): ").strip()
        source = file_path
        print(f"Opening video file: {file_path}")
    elif choice == "3":
        ip_url = input("Enter Phone IP Webcam URL (e.g., http://192.168.1.5:8080/video): ").strip()
        source = ip_url
        print(f"Connecting to Phone Camera at {ip_url}...")
    else:
        print("Invalid choice! Defaulting to Laptop Webcam.")
        source = 0

    # Load YOLOv8 nano model
    model = YOLO('yolov8n.pt')
    
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("Error: Could not open the selected video source!")
        return

    print("\nAI Detection Running! Press 'q' on your keyboard inside the video window to stop.\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video stream ended or frame could not be read.")
            break

        # Run YOLO detection on the frame
        results = model(frame)
        annotated_frame = results[0].plot()
        
        # Count vehicles detected in current frame
        vehicle_count = len(results[0].boxes)

        # Display vehicle count on top of the video window
        cv2.putText(annotated_frame, f"Live Vehicle Count: {vehicle_count}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show live video window
        cv2.imshow("AI Smart Traffic Live Detection", annotated_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_detection()