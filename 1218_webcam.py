import cv2
import mediapipe as mp
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import time


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

# 프레임 리사이즈 및 중앙 크롭 함수
def resize_to_fit_window(frame, target_width, target_height):
    frame_height, frame_width = frame.shape[:2]
    frame_aspect_ratio = frame_width / frame_height
    target_aspect_ratio = target_width / target_height

    if frame_aspect_ratio > target_aspect_ratio:
        # 프레임이 더 넓을 때: 너비를 줄이고 높이를 맞춤
        new_height = target_height
        new_width = int(new_height * frame_aspect_ratio)
    else:
        # 프레임이 더 좁을 때: 높이를 줄이고 너비를 맞춤
        new_width = target_width
        new_height = int(new_width / frame_aspect_ratio)

    # 프레임 리사이즈
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 중앙 크롭
    x_start = (new_width - target_width) // 2
    y_start = (new_height - target_height) // 2
    cropped_frame = resized_frame[y_start:y_start + target_height, x_start:x_start + target_width]

    return cropped_frame


# 프레임 오버레이 함수
def overlay_frames(base_frame, overlay_frame, alpha=0.5):
    return cv2.addWeighted(base_frame, 1, overlay_frame, 1, 0)

# Mediapipe Pose 모델 초기화
mp_pose = mp.solutions.pose

###### 추 가 
# Mediapipe Pose 및 Drawing Utilities 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose




# 실시간 웹캠 입력과 전문가 포즈 비교 및 동영상 오버레이
def process_webcam_with_overlay_and_compare(expert_keypoints_list, overlay_video_path):
    # 전문가 키포인트 미리 추출
    normalized_expert_keypoints_list = [normalize_keypoints(kp) for kp in expert_keypoints_list]
    num_expert_frames = len(normalized_expert_keypoints_list)

    # 웹캠 및 동영상 열기
    cap_webcam = cv2.VideoCapture(0)
    cap_overlay = cv2.VideoCapture(overlay_video_path)

    cv2.setUseOptimized(True)  # OpenCV 최적화 활성화
    cv2.setNumThreads(4)       # 사용할 스레드 수 설정 (시스템에 맞게 조정)
    
    # bigoutput.mp4의 FPS 가져오기
    overlay_fps = cap_overlay.get(cv2.CAP_PROP_FPS)
    webcam_fps = cap_webcam.get(cv2.CAP_PROP_FPS)
    print(f"Overlay Video FPS: {overlay_fps}")
    print(f"Webcam FPS: {webcam_fps}")

    # FPS가 유효하지 않을 경우 기본값 설정
    if overlay_fps == 0:
        overlay_fps = 30.0
    if webcam_fps == 0:
        webcam_fps = 30.0

    # 동기화를 위한 타겟 FPS 설정 (높은 쪽에 맞춤)
    target_fps = max(overlay_fps, webcam_fps)
    frame_delay = int(1000 / target_fps)  # 밀리초 단위의 프레임 간 대기 시간

    
    prev_time = time.time()  # 초기값 설정
    score = 0
    feedback="Waiting for Pose Detection..."

    # 화면 설정
    window_width, window_height = 607, 1080
    frame_idx = 0

    
    movement_threshold = 0.02  # 움직임을 감지하는 임계값 (조정 가능)
    stillness_frames = 0       # 사용자가 가만히 있는 프레임 수
    
    # 이전 키포인트 초기화
    prev_user_keypoints = None

    # 실루엣 맞춤 상태 플래그
    # ready_to_start = False
    
    with mp_pose.Pose() as pose:
        while cap_webcam.isOpened():
            ret_webcam, frame_webcam = cap_webcam.read()
            ret_overlay, frame_overlay = cap_overlay.read()
    
            if not ret_webcam or not ret_overlay:
                break
    
            # 웹캠 프레임 처리
            frame_webcam = cv2.flip(frame_webcam, 1)
            frame_webcam_resized = resize_to_fit_window(frame_webcam, window_width, window_height)
            frame_overlay_resized = resize_to_fit_window(frame_overlay, window_width, window_height)
            combined_frame = overlay_frames(frame_webcam_resized, frame_overlay_resized, alpha=0.5)

            # 사용자 포즈 추출
            frame_rgb = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB)
            result_webcam = pose.process(frame_rgb)
    
            if result_webcam.pose_landmarks:
                # 사용자 키포인트 추출 및 정규화
                user_keypoints = np.array([[lmk.x, lmk.y, lmk.z] for lmk in result_webcam.pose_landmarks.landmark])
                normalized_user_keypoints = normalize_keypoints(user_keypoints)

                ########추 가 ....
                
                # 포즈 랜드마크 시각화
                mp_drawing.draw_landmarks(frame_webcam, result_webcam.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                #####################

            
                if prev_user_keypoints is not None:
                    # 움직임 계산
                    movement = np.linalg.norm(normalized_user_keypoints - prev_user_keypoints)
    
                    if movement < movement_threshold:
                        # 움직임이 없으면 경고 메시지 출력
                        stillness_frames += 1
                        if stillness_frames > 3:  # 3프레임 동안 가만히 있으면 메시지 출력
                            cv2.putText(combined_frame, "Move your body!", (50, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.putText(combined_frame, feedback, (50, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                    else:
                        # 움직임이 감지되면 점수 평가
                        stillness_frames = 0  # 가만히 있는 프레임 수 리셋
    
                        # 전문가 프레임과 비교 (순환 재생)
                        normalized_expert_keypoints = normalized_expert_keypoints_list[frame_idx % num_expert_frames]
    
                        # 1D로 변환 후 DTW 거리 계산
                        expert_flat = normalized_expert_keypoints.flatten()
                        user_flat = normalized_user_keypoints.flatten()
                        distance, _ = fastdtw(expert_flat.reshape(-1, 1), user_flat.reshape(-1, 1), dist=euclidean)
    
                        
                        # 점수 계산
                        max_distance = 5.5
                        score = max(100 - (distance / max_distance) * 100, 0)
        
                        # 점수에 따른 피드백
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
    
                        # 점수와 피드백 출력
                        cv2.putText(combined_frame, f"Score: {score:.2f}", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(combined_frame, feedback, (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
                # 현재 프레임 키포인트를 이전 키포인트로 저장
                prev_user_keypoints = normalized_user_keypoints
            else:
                # 포즈가 감지되지 않으면 안내 메시지 출력
                cv2.putText(combined_frame, "No pose detected. Stand in front of the camera.",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
            # 화면 출력
            cv2.imshow("Webcam with Video Overlay & Pose Comparison", combined_frame)

            #############

            # 화면에 프레임 출력
            cv2.imshow("Pose Detection with Overlay", frame_webcam)
    
            ################
    
            # 프레임 인덱스 업데이트
            frame_idx += 1
    
            # 종료 조건
            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                break
    
        cap_webcam.release()
        cap_overlay.release()
        cv2.destroyAllWindows()

 # 전문가 영상 경로 및 오버레이 동영상 경로
expert_video_path = "videos1/expert_dance1.mp4"
overlay_video_path = "videos1/bigoutput.mp4"

# 전문가 키포인트 추출
print("Extracting expert keypoints...")
expert_keypoints_list = extract_keypoints_from_video(expert_video_path)

# 함수 실행
process_webcam_with_overlay_and_compare(expert_keypoints_list, overlay_video_path)
         
