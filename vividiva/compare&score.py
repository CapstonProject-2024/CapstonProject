import cv2
import mediapipe as mp
import numpy as np
import time

# Mediapipe Pose 모델 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 포즈 정규화 함수 (기준점: 골반)
def normalize_keypoints(keypoints):
    pelvis = keypoints[24]  # 골반 기준
    return np.nan_to_num(keypoints - pelvis)

# 유클리드 거리 기반 유사도 계산 함수
def calculate_similarity(expert_keypoints, user_keypoints):
    distances = np.linalg.norm(expert_keypoints - user_keypoints, axis=1)  # 각 키포인트 간 거리 계산
    mean_distance = np.mean(distances)  # 평균 거리
    max_distance = 0.5  # 유사도 점수 변환 기준 (조정 가능)
    score = max(100 - (mean_distance / max_distance) * 100, 0)  # 100점 만점으로 변환
    return score

# 전문가 영상과 사용자의 웹캠 비교
def process_webcam_and_compare_with_expert(expert_video_path):
    cap_expert = cv2.VideoCapture(expert_video_path)
    cap_webcam = cv2.VideoCapture(0)

    expert_width = int(cap_expert.get(cv2.CAP_PROP_FRAME_WIDTH))
    expert_height = int(cap_expert.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expert_fps = cap_expert.get(cv2.CAP_PROP_FPS) or 30

    frame_delay = int(1000 / expert_fps)
    frame_idx = 0
    last_calculation_time = 0  # 마지막 계산 시간
    score = 0  # 초기 점수

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap_expert.isOpened() and cap_webcam.isOpened():
            ret_expert, frame_expert = cap_expert.read()
            ret_webcam, frame_webcam = cap_webcam.read()

            if not ret_expert:
                cap_expert.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 비디오 끝나면 처음으로 돌아감
                continue
            if not ret_webcam:
                print("Webcam frame not available")
                break

            # 웹캠 좌우 반전 및 해상도 조정
            frame_webcam = cv2.flip(frame_webcam, 1)
            frame_webcam_resized = cv2.resize(frame_webcam, (expert_width, expert_height))

            # Mediapipe 적용
            result_expert = pose.process(cv2.cvtColor(frame_expert, cv2.COLOR_BGR2RGB))
            result_webcam = pose.process(cv2.cvtColor(frame_webcam_resized, cv2.COLOR_BGR2RGB))

            # 1초마다 점수 계산
            current_time = time.time()
            if current_time - last_calculation_time >= 0.5:  # 1초 경과
                if result_expert.pose_landmarks and result_webcam.pose_landmarks:
                    expert_keypoints = np.array([[lmk.x, lmk.y, lmk.z] for lmk in result_expert.pose_landmarks.landmark])
                    user_keypoints = np.array([[lmk.x, lmk.y, lmk.z] for lmk in result_webcam.pose_landmarks.landmark])

                    if expert_keypoints.shape == (33, 3) and user_keypoints.shape == (33, 3):
                        normalized_expert = normalize_keypoints(expert_keypoints)
                        normalized_user = normalize_keypoints(user_keypoints)
                        score = calculate_similarity(normalized_expert, normalized_user)  # 유사도 계산

                last_calculation_time = current_time  # 마지막 계산 시간 업데이트

            # 화면에 점수 출력
            cv2.putText(frame_webcam_resized, f"Score: {score:.2f}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Mediapipe 포즈 시각화
            mp_drawing.draw_landmarks(frame_expert, result_expert.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame_webcam_resized, result_webcam.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 두 프레임 결합
            combined_frame = cv2.hconcat([frame_expert, frame_webcam_resized])
            cv2.imshow("Expert Video vs Webcam Pose Comparison", combined_frame)

            # 종료 조건
            if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
                break

            frame_idx += 1

    cap_expert.release()
    cap_webcam.release()
    cv2.destroyAllWindows()

# 전문가 영상 경로 설정
expert_video_path = "vividiva.mp4"

# 실행
process_webcam_and_compare_with_expert(expert_video_path)
