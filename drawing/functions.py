# 전문가 키포인트 추출
def extract_keypoints_from_video(video_path):
    keypoints_list = []
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    with mp_pose.Pose() as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                keypoints = np.array([[lmk.x, lmk.y, lmk.z] for lmk in result.pose_landmarks.landmark])
                keypoints_list.append(keypoints)
    cap.release()
    return np.array(keypoints_list)


# 포즈 정규화 함수 (기준점: 골반)
def normalize_keypoints(keypoints):
    pelvis = keypoints[24]  # 골반 좌표 (Assuming landmark 24 is pelvis)
    return keypoints - pelvis
    

def save_keypoints_to_file(keypoints, file_path):
    np.save(file_path, keypoints)
    

def load_keypoints_from_file(file_path):
    return np.load(file_path)


# 키포인트 비교 -> 동작 비교
def calculate_similarity(expert_keypoints, user_keypoints):
    """Calculate similarity using DTW."""
    # 1-D 배열로 변환 및 데이터 타입 강제 변환
    expert_flat = expert_keypoints.flatten()
    user_flat = user_keypoints.flatten()

    # 차원 확인
    print(f"Expert_flat: {expert_flat.shape}, dtype: {expert_flat.dtype}")
    print(f"User_flat: {user_flat.shape}, dtype: {user_flat.dtype}")

    # DTW 거리 계산
    try:
        distance = dtw.distance(expert_flat, user_flat)
        #distance, _ = fastdtw(expert_flat, user_flat, dist=euclidean)
        print(f"DTW Distance: {distance}")
        max_distance = 5.5
        return max(100 - (distance / max_distance) * 100, 0)
    except Exception as e:
        print(f"Error in DTW calculation: {e}")
        return 0.0
