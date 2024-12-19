import cv2
import os

# 비디오 경로와 저장 디렉토리 설정
video_path = "/content/drive/MyDrive/Colab Notebooks/vividiva/vividiva.mp4"
output_dir = "/content/drive/MyDrive/Colab Notebooks/vividiva/test"
os.makedirs(output_dir, exist_ok=True)

# 비디오 읽기
cap = cv2.VideoCapture(video_path)
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 저장
    frame_path = os.path.join(output_dir, f"{frame_idx:04d}.jpg")
    cv2.imwrite(frame_path, frame)
    frame_idx += 1

cap.release()
print(f"저장 완료! 총 {frame_idx}개의 프레임이 저장되었습니다.")
