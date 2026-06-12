import cv2
import time
from segmentation_utils import segment_and_colorize

video_path = 0  # 0 means webcam
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video source")
    exit()

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    colorized, mask = segment_and_colorize(frame)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(colorized, f"FPS: {fps:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Original Video", frame)
    cv2.imshow("Colorized Output", colorized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()