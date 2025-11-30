import cv2
import ultralytics


model = ultralytics.YOLO('yolov8s.pt')

cap = cv2.VideoCapture('meetings.mp4')

while True:
    success, frame = cap.read()

    if not success:
        break

    if cv2.waitKey(20) & 0xFF == 27:
        break

    frame = cv2.resize(frame, None, fx=0.2, fy=0.2)

    results = model.predict(frame, conf=0.25, iou=0.6, classes=[0])
    result = results[0]
    people_counter = len(result.boxes.cls)

    result_frame = result.plot()

    if people_counter == 5:
        cv2.imshow('orig', frame)
        cv2.imshow('res', result_frame)

cap.release()