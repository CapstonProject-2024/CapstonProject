# Mediapipe Pose 모델 초기화
mp_pose = mp.solutions.pose


# 실시간 웹캠 입력과 전문가 포즈 비교 및 동영상 오버레이
def process_webcam_with_overlay_and_compare(expert_keypoints_list, overlay_video_path):
    # 전문가 키포인트 미리 추출
    normalized_expert_keypoints_list = [normalize_keypoints(kp) for kp in expert_keypoints_list]
    num_expert_frames = len(normalized_expert_keypoints_list)

    # 웹캠 및 동영상 열기
    cap_webcam = cv2.VideoCapture(0)
    cap_overlay = cv2.VideoCapture(overlay_video_path)

    # bigoutput.mp4의 FPS 가져오기
    overlay_fps = cap_overlay.get(cv2.CAP_PROP_FPS)
    print(overlay_fps)
    if overlay_fps == 0:  # FPS 정보가 없으면 기본값 설정
        overlay_fps = 24.0
    frame_delay = int(1000 / overlay_fps)  # 밀리초 단위의 프레임 간 대기 시간
    prev_time = time.time()  # 초기값 설정
    score = 0
    feedback="Waiting for Pose Detection..."

    # 화면 설정
    window_width, window_height = 607, 1080
    frame_idx = 0

    with mp_pose.Pose() as pose:
        while cap_webcam.isOpened():
            ret_webcam, frame_webcam = cap_webcam.read()
            ret_overlay, frame_overlay = cap_overlay.read()

            if not ret_webcam:
                print("Error: Webcam frame cannot be read.")
                break

            if not ret_overlay:  # 비디오 끝나면 되감기
                cap_overlay.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # 웹캠 및 동영상 리사이ㅇ
            frame_webcam = cv2.flip(frame_webcam, 1)
            frame_webcam_resized = resize_to_fit_window(frame_webcam, window_width, window_height)
            frame_overlay_resized = resize_to_fit_window(frame_overlay, window_width, window_height)

            # 웹캠에 동영상 오버레이
            combined_frame = overlay_frames(frame_webcam_resized, frame_overlay_resized, alpha=0.5)

            # 사용자 포즈 추출
            frame_rgb = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB)
            result_webcam = pose.process(frame_rgb)

            if result_webcam.pose_landmarks:
                # 사용자 키포인트 추출 및 정규화
                user_keypoints = np.array([[lmk.x, lmk.y, lmk.z] for lmk in result_webcam.pose_landmarks.landmark])
                normalized_user_keypoints = normalize_keypoints(user_keypoints)
                
                # 전문가 프레임과 비교 (순환 재생)
                normalized_expert_keypoints = normalized_expert_keypoints_list[frame_idx % num_expert_frames]
                
                # 1D로 변환 후 DTW 거리 계산
                expert_flat = normalized_expert_keypoints.flatten()
                user_flat = normalized_user_keypoints.flatten()
                distance, _ = fastdtw(expert_flat.reshape(-1, 1), user_flat.reshape(-1, 1), dist=euclidean)
                
                # 점수 계산 (100점 만점 변환)
                max_distance = 40  # 'bad' 기준인 40을 최대 거리로 설정
                score = max(100 - (distance / max_distance) * 100, 0)  # 100에서 거리 비율만큼 감소
                
                # 평가 점수 및 멘트
                if score >= 95:
                    feedback = "Perfect! Great job!"
                elif score >= 85:
                    feedback = "Good! You're almost there!"
                elif score >= 75:
                    feedback = "Normal! Keep going!"
                elif score >= 60:
                    feedback = "Nice try! You're getting there!"
                else:
                    feedback = "Good effort! Keep it up!"

            # 화면에 텍스트 출력
            cv2.putText(combined_frame, f"Score: {score:.2f}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined_frame, feedback, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
            # 프레임 인덱스 업데이트
            frame_idx += 1

                # 화면 출력
            cv2.imshow("Webcam with Video Overlay & Pose Comparison", combined_frame)



            
            # FPS에 맞춘 대기 시간 적용
            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                break

    cap_webcam.release()
    cap_overlay.release()
    cv2.destroyAllWindows()


 # 전문가 영상 경로 및 오버레이 동영상 경로
expert_video_path = "mantra.mp4"
overlay_video_path = "bigoutput.mp4"

# 전문가 키포인트 추출
print("Extracting expert keypoints...")
expert_keypoints_list = extract_keypoints_from_video(expert_video_path)

# 함수 실행
process_webcam_with_overlay_and_compare(expert_keypoints_list, overlay_video_path)
                      
        
