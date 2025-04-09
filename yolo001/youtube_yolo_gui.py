import tkinter as tk
from tkinter import messagebox
import cv2
import torch
import yt_dlp

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 유튜브 스트리밍 URL 가져오기 함수
def get_streaming_url(video_url):
    try:
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            return info['url']
    except Exception as e:
        messagebox.showerror("Error", f"유튜브 스트리밍 URL을 가져오는 중 오류가 발생했습니다.\n{e}")
        return None

# 객체 인식 함수
def detect_objects(video_url):
    try:
        streaming_url = get_streaming_url(video_url)
        if not streaming_url:
            return

        cap = cv2.VideoCapture(streaming_url)  # 스트리밍 URL로 OpenCV 비디오 캡처

        if not cap.isOpened():
            messagebox.showerror("Error", "유튜브 동영상을 열 수 없습니다.")
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

    except Exception as e:
        messagebox.showerror("Error", f"유튜브 동영상을 처리하는 중 오류가 발생했습니다.\n{e}")

# GUI 생성
def start_detection():
    video_url = url_entry.get()
    if not video_url:
        messagebox.showerror("Error", "유튜브 동영상 URL을 입력하세요.")
        return

    detect_objects(video_url)

root = tk.Tk()
root.title("YOLOv5 YouTube Object Detection")

url_label = tk.Label(root, text="유튜브 동영상 URL:", font=("Arial", 12))
url_label.pack(pady=5)

url_entry = tk.Entry(root, width=50, font=("Arial", 12))
url_entry.pack(pady=5)

start_button = tk.Button(root, text="시작", command=start_detection, font=("Arial", 14), bg="green", fg="white")
start_button.pack(pady=20)

info_label = tk.Label(root, text="유튜브 동영상 URL을 입력하고 시작 버튼을 누르세요.\n종료하려면 'q'를 누르세요.", font=("Arial", 12))
info_label.pack(pady=10)

root.mainloop()