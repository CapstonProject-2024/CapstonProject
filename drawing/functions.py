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



# 웹캠 오버레이
def webcam_with_overlay_and_comparison(audio_file, overlay_video, expert_keypoints, stop_event):
    cap_webcam = cv2.VideoCapture(0)
    cap_overlay = cv2.VideoCapture(overlay_video)

    if not cap_webcam.isOpened():
        print("Error: Cannot access webcam.")
        stop_event.set()
        return

    if not cap_overlay.isOpened():
        print("Error: Cannot access overlay video.")
        stop_event.set()
        return

    # 해상도 설정
    width, height = 607, 1080
    cap_webcam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap_webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    fps_overlay = cap_overlay.get(cv2.CAP_PROP_FPS)
    frame_interval = 1 / fps_overlay if fps_overlay > 0 else 0.033

    normalized_expert_keypoints_list = [normalize_keypoints(kp) for kp in expert_keypoints]
    num_expert_frames = len(normalized_expert_keypoints_list)

    print("Webcam and countdown starting...")

    # 카운트다운
    countdown_start_time = time.time()
    countdown_duration = 10  # 10초 카운트다운

    while time.time() - countdown_start_time < countdown_duration:
        ret_webcam, frame_webcam = cap_webcam.read()
        if not ret_webcam:
            print("Failed to capture webcam frame during countdown.")
            stop_event.set()
            break

        # 거울 모드로 변경
        frame_webcam = cv2.flip(frame_webcam, 1)

        frame_webcam_resized = cv2.resize(frame_webcam, (width, height))
        countdown_text = f"Starting in {int(countdown_duration - (time.time() - countdown_start_time))} seconds..."
        cv2.putText(frame_webcam_resized, countdown_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Dance App", frame_webcam_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    # 오디오 재생
    print("Playing audio and starting overlay.")
    audio_process = subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", audio_file],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # 오버레이 및 동작 비교
    start_time = time.time()
    frame_index = 0
    score = 0

    with mp.solutions.pose.Pose() as pose:
        while not stop_event.is_set():
            elapsed_time = time.time() - start_time

            ret_webcam, frame_webcam = cap_webcam.read()
            if not ret_webcam:
                print("Failed to capture webcam frame during overlay.")
                stop_event.set()
                break

            
            # 거울 모드로 변경
            frame_webcam = cv2.flip(frame_webcam, 1)

            cap_overlay.set(cv2.CAP_PROP_POS_MSEC, elapsed_time * 1000)
            ret_overlay, frame_overlay = cap_overlay.read()
            if not ret_overlay:
                print("Overlay video ended.")
                break

            frame_webcam_resized = cv2.resize(frame_webcam, (width, height))
            frame_overlay_resized = cv2.resize(frame_overlay, (width, height))
            combined_frame = cv2.addWeighted(frame_webcam_resized, 1, frame_overlay_resized, 1, 0)

            result = pose.process(cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks and frame_index < len(expert_keypoints):
                user_keypoints = np.array([[lmk.x, lmk.y, lmk.z] for lmk in result.pose_landmarks.landmark])
                expert_keypoints_frame = expert_keypoints[frame_index]
                
                normalized_user_keypoints = normalize_keypoints(user_keypoints)
                normalized_expert_keypoints = normalized_expert_keypoints_list[frame_index % num_expert_frames]
                
                score = calculate_similarity(normalized_expert_keypoints, normalized_user_keypoints)
            
            cv2.putText(combined_frame, f"Score: {score:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Dance App", combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

            frame_index += 1
            time.sleep(frame_interval)

    cap_webcam.release()
    cap_overlay.release()
    cv2.destroyAllWindows()
    audio_process.terminate()
