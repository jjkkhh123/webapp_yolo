import tkinter as tk
from tkinter import messagebox
import cv2
import torch
import numpy as np

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 객체 인식 함수
def detect_objects():
    cap = cv2.VideoCapture(0)  # 웹캠 열기
    if not cap.isOpened():
        messagebox.showerror("Error", "웹캠을 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv5 모델로 추론
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()  # 바운딩 박스, 클래스, confidence

        # 바운딩 박스 그리기
        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 화면에 출력
        cv2.imshow("YOLOv5 Object Detection", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI 생성
def start_detection():
    detect_objects()

root = tk.Tk()
root.title("YOLOv5 Object Detection")

start_button = tk.Button(root, text="시작", command=start_detection, font=("Arial", 14), bg="green", fg="white")
start_button.pack(pady=20)

info_label = tk.Label(root, text="시작 버튼을 눌러 웹캠 객체 인식을 시작하세요.\n종료하려면 'q'를 누르세요.", font=("Arial", 12))
info_label.pack(pady=10)

root.mainloop()