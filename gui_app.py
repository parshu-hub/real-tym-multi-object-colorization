import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import time
from segmentation_utils import segment_and_colorize

def run_video(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video source")
        return

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        colorized, _ = segment_and_colorize(frame)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(colorized, f"FPS: {fps:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Original Video", frame)
        cv2.imshow("Colorized Output", colorized)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def open_webcam():
    run_video(0)

def open_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    if file_path:
        run_video(file_path)

root = tk.Tk()
root.title("Real-Time Multi-Object Colorization")
root.geometry("400x250")

label = tk.Label(root, text="Real-Time Multi-Object Colorization", font=("Arial", 14, "bold"))
label.pack(pady=20)

btn_webcam = tk.Button(root, text="Use Webcam", width=20, command=open_webcam)
btn_webcam.pack(pady=10)

btn_video = tk.Button(root, text="Upload Video", width=20, command=open_video)
btn_video.pack(pady=10)

btn_exit = tk.Button(root, text="Exit", width=20, command=root.destroy)
btn_exit.pack(pady=10)

root.mainloop()