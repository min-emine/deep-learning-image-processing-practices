from ultralytics import YOLO
import cv2

model=YOLO("yolov8n.pt")

video_path = "IMG_5268.MOV"
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out=cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    
    
    results = model.track(
        frame,
        persist=True,
        conf=0.3,
        iou=0.5,
        tracker="bytetrack.yaml",
        classes=[2]

    )
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Tracking", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()