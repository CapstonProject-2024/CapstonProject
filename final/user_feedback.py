import cv2
import mediapipe as mp
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import time



# Mediapipe Pose 모델 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# OpenCV 최적화 설정
cv2.setUseOptimized(True)
cv2.setNumThreads(4)


# 고정된 크기로 프레임 리사이즈 함수
def resize_frame_fixed_size(frame1, frame2, width=640, height=480):
    frame1_resized = cv2.resize(frame1, (width, height))
    frame2_resized = cv2.resize(frame2, (width, height))
    return frame1_resized, frame2_resized
    
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
    pelvis = keypoints[24]  # 골반 좌표
    return keypoints - pelvis


def process_and_compare_videos(expert_video_path, amateur_video_path):

    # 전문가 키포인트 추출 및 정규화
    expert_keypoints_list = extract_keypoints_from_video(expert_video_path)
    normalized_expert_keypoints_list = [normalize_keypoints(kp) for kp in expert_keypoints_list]

    # 전문가 및 사용자 영상 캡처 객체 초기화
    cap_expert = cv2.VideoCapture(expert_video_path)
    cap_amateur = cv2.VideoCapture(amateur_video_path)

    frame_idx = 0
    
    with mp_pose.Pose() as pose:
        while cap_expert.isOpened() and cap_amateur.isOpened():
            ret_expert, frame_expert = cap_expert.read()
            ret_amateur, frame_amateur = cap_amateur.read()

            if not ret_expert or not ret_amateur:
                break

            # 프레임 리사이즈
            frame_expert_resized, frame_amateur_resized = resize_frame_fixed_size(frame_expert, frame_amateur)

            frame_amateur_rgb = cv2.cvtColor(frame_amateur_resized, cv2.COLOR_BGR2RGB)
            result_amateur = pose.process(frame_amateur_rgb)


            # 현재 시간 확인
            current_time = time.time()

            # 사용자 키포인트 추출 
            if result_amateur.pose_landmarks:
                amateur_keypoints = np.array([[lmk.x, lmk.y, lmk.z] for lmk in result_amateur.pose_landmarks.landmark])
                normalized_amateur_keypoints = normalize_keypoints(amateur_keypoints)
            
                # 전문가 키포인트와 비교
                normalized_expert_keypoints = normalized_expert_keypoints_list[frame_idx % len(normalized_expert_keypoints_list)]
                
                # 1차원 벡터로 변환 및 2차원 형태로 재구성
                expert_keypoints_flat = normalized_expert_keypoints.flatten().reshape(-1, 1)
                amateur_keypoints_flat = normalized_amateur_keypoints.flatten().reshape(-1, 1)

                # 키포인트 길이 맞추기 (필요 시)
                min_length = min(len(expert_keypoints_flat), len(amateur_keypoints_flat))
                expert_keypoints_flat = expert_keypoints_flat[:min_length]
                amateur_keypoints_flat = amateur_keypoints_flat[:min_length]
            
                # DTW 거리 계산
                distance, _ = fastdtw(expert_keypoints_flat, amateur_keypoints_flat, dist=euclidean)

                # 점수 계산 및 출력
                max_distance = 7
                # 점수 계산식에 완화 계수 적용
                sensitivity = 0.3  # 0.0 ~ 1.0 사이 (낮을수록 관대함)
                score = max(100 - (distance / max_distance) * 100 * sensitivity, 0)
                
                cv2.putText(frame_amateur_resized, f"Score: {score:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 점수와 피드백 저장
                last_score = f"Score: {score:.2f}"
                if score >= 95:
                    last_feedback = "Perfect! Great job!"
                elif score >= 85:
                    last_feedback = "Good! You're almost there!"
                elif score >= 75:
                    last_feedback = "Normal! Keep going!"
                elif score >= 60:
                    last_feedback = "Nice try! You're getting there!"
                elif score > 15:
                    last_feedback = "Good effort! Keep it up!"
                else:
                    last_feedback = "It's Okay... Don't give up!"
                
                # 마지막 점수 계산 시간 업데이트
                last_score_time = current_time

            # 점수와 피드백을 계속 표시
            cv2.putText(frame_amateur_resized, last_score, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_amateur_resized, last_feedback, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 두 영상 나란히 보기
            combined_frame = cv2.hconcat([frame_expert_resized, frame_amateur_resized])

            # 결합된 프레임 출력
            cv2.imshow("Amateur Comparison", combined_frame)

            # 종료 조건 (q 키를 누르면 종료)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1


    cap_expert.release()
    cap_amateur.release()
    cv2.destroyAllWindows()
    
# 전문가 영상과 사용자 영상 경로 설정
expert_video_path = "videos1/expert_dance1.mp4"
amateur_video_path = "videos1/amateur_dance1.mp4"

# 함수 실행
process_and_compare_videos(expert_video_path, amateur_video_path)

