# 전문가 영상에서 키포인트 추출 함수
def extract_keypoints_from_video(video_path):
    keypoints_list = []
    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose() as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)
            if result.pose_landmarks:
                keypoints = np.array([[lmk.x, lmk.y, lmk.z] for lmk in result.pose_landmarks.landmark])
                keypoints_list.append(keypoints)
    cap.release()
    return np.array(keypoints_list)



# 포즈 정규화 함수 (기준점: 골반)
def normalize_keypoints(keypoints):
    pelvis = keypoints[24]  # 골반 좌표 (Assuming landmark 24 is pelvis)
    return keypoints - pelvis

###############################################################################

# 프레임 리사이즈 함수
def resize_to_fit_window(frame, target_width, target_height):
    frame_height, frame_width = frame.shape[:2]
    aspect_ratio = frame_width / frame_height

    if target_width / target_height > aspect_ratio:
        new_height = target_height
        new_width = int(aspect_ratio * new_height)
    else:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)

    resized_frame = cv2.resize(frame, (new_width, new_height))
    result_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    result_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
    return result_frame

# 프레임 오버레이 함수
def overlay_frames(base_frame, overlay_frame, alpha=0.5):
    return cv2.addWeighted(base_frame, 1, overlay_frame, 1, 0)
