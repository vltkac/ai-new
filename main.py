import cv2
import numpy as np
from ultralytics import YOLO


def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def kp(xy, idx):
    return xy[0, idx]


video = cv2.VideoCapture('data/lesson_pose/squat.mp4')
model = YOLO('yolo11s-pose.pt')

move_down = True
counter = 0

while video.isOpened():
    ok, frame = video.read()
    if not ok:
        break

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    result = model.predict(frame, verbose=False)[0]
    frame = result.plot()

    if result.keypoints is None:
        cv2.imshow("Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    xy = result.keypoints.xy

    hip = kp(xy, 12)
    knee = kp(xy, 14)
    ankle = kp(xy, 16)

    angle = calculate_angle(hip, knee, ankle)

    if angle < 70 and move_down:
        counter += 1
        move_down = False

    if angle > 160 and not move_down:
        move_down = True

    cv2.putText(
        frame,
        f"Squats: {counter}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Pose", frame)
    print(int(angle))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()