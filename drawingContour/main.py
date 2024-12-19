# 비디오 파일 열기
video_path = "/content/drive/MyDrive/Colab Notebooks/vividiva/segmentation.mp4"  # 원본 비디오 경로
cap = cv2.VideoCapture(video_path)

# 비디오 프레임 크기와 FPS 정보 가져오기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 비디오 저장을 위한 설정
output_video_path = "/content/drive/MyDrive/Colab Notebooks/vividiva/contour.mp4"  # 저장할 비디오 경로

# VideoWriter 객체 생성 (XVID 코덱을 사용하여 mp4 파일로 저장)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# 비디오 프레임 처리
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 비디오 프레임 개수

for frame_idx in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break  # 더 이상 읽을 프레임이 없으면 종료

    # Segmentation 마스크를 프레임에 추가 (투명 배경으로 설정)
    overlay_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # 'video_segments'에서 해당 프레임의 마스크를 가져와서 외곽선 그리기
    if frame_idx in video_segments:  # 비디오 세그멘테이션이 존재하는 경우에만 처리
        for out_obj_id, out_mask in video_segments[frame_idx].items():
            tmp_mask = out_mask.squeeze()  # 마스크 차원 조정
            # 외곽선 그리기
            overlay_image = draw_neon_contours(
                overlay_image, tmp_mask, neon_color=(255, 0, 255), thickness=2
            )

    # 이미지를 BGR로 변환 (OpenCV는 BGR 포맷을 사용)
    overlay_image_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)

    # 비디오에 프레임 추가
    out.write(overlay_image_bgr)

# 비디오 파일 저장 종료
out.release()
cap.release()
