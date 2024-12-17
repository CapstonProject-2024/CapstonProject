import cv2
import mediapipe as mp
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
import time

# Mediapipe Pose 모델 초기화
mp_pose = mp.solutions.pose

# 포즈 추출 및 실시간 비교 함수
def process_and_compare_videos(expert_video_path, amateur_video_path, target_fps=30):
    cap_expert = cv2.VideoCapture(expert_video_path)
    cap_amateur = cv2.VideoCapture(amateur_video_path)

    # 두 영상의 FPS를 동일하게 설정
    cap_expert.set(cv2.CAP_PROP_FPS, target_fps)
    cap_amateur.set(cv2.CAP_PROP_FPS, target_fps)

    # 프레임 크기 조정 함수
    def resize_frame_to_smallest_height(frame1, frame2):
        # 두 프레임의 높이 중 작은 값으로 맞춤
        min_height = min(frame1.shape[0], frame2.shape[0])
        aspect_ratio1 = frame1.shape[1] / frame1.shape[0]
        aspect_ratio2 = frame2.shape[1] / frame2.shape[0]
        frame1_resized = cv2.resize(frame1, (int(min_height * aspect_ratio1), min_height))
        frame2_resized = cv2.resize(frame2, (int(min_height * aspect_ratio2), min_height))
        return frame1_resized, frame2_resized
    
    with mp_pose.Pose() as pose:
        while cap_expert.isOpened() and cap_amateur.isOpened():
            ret_expert, frame_expert = cap_expert.read()
            ret_amateur, frame_amateur = cap_amateur.read()

            # 프레임 읽기 실패 시 처리
            if frame_expert is None or frame_amateur is None:
                print("Error: One of the frames could not be read properly.")
                break
                
            if not ret_expert or not ret_amateur:
                break
            
            # RGB로 변환
            frame_expert_rgb = cv2.cvtColor(frame_expert, cv2.COLOR_BGR2RGB)
            frame_amateur_rgb = cv2.cvtColor(frame_amateur, cv2.COLOR_BGR2RGB)
            
            # Mediapipe로 포즈 추출
            result_expert = pose.process(frame_expert_rgb)
            result_amateur = pose.process(frame_amateur_rgb)
            
            if result_expert.pose_landmarks and result_amateur.pose_landmarks:
                # 키포인트 추출
                expert_keypoints = np.array([[lmk.x, lmk.y, lmk.z] for lmk in result_expert.pose_landmarks.landmark])
                amateur_keypoints = np.array([[lmk.x, lmk.y, lmk.z] for lmk in result_amateur.pose_landmarks.landmark])
                
                # 포즈 정규화 (골반 기준)
                pelvis_expert = expert_keypoints[24]
                pelvis_amateur = amateur_keypoints[24]
                
                normalized_expert = expert_keypoints - pelvis_expert
                normalized_amateur = amateur_keypoints - pelvis_amateur
                
                # 1D로 변환 후 DTW 거리 계산
                expert_flat = normalized_expert.flatten()
                amateur_flat = normalized_amateur.flatten()
                distance, _ = fastdtw(expert_flat.reshape(-1, 1), amateur_flat.reshape(-1, 1), dist=euclidean)
                
                """
                # 평가 점수 및 멘트
                if distance < 10:
                    feedback = "Perfect! Great job!"
                elif distance < 30:
                    feedback = "Good job! You'll be perfect!"
                else:
                    feedback = "It's okay! Keep trying!"
                
                # 화면에 텍스트 출력
                cv2.putText(frame_amateur, f"Score: {distance:.2f}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame_amateur, feedback, (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                """
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
                cv2.putText(frame_amateur, f"Score: {score:.2f}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame_amateur, feedback, (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            
            # 두 영상의 프레임을 동일한 높이로 맞춤
            frame_expert_resized, frame_amateur_resized = resize_frame_to_smallest_height(frame_expert, frame_amateur)

            # 데이터 타입 맞춤 (uint8로 변환)
            frame_expert_resized = frame_expert_resized.astype(np.uint8)
            frame_amateur_resized = frame_amateur_resized.astype(np.uint8)

            # 두 영상 나란히 보기
            combined_frame = cv2.hconcat([frame_expert_resized, frame_amateur_resized])
            
            # 결합된 프레임에 텍스트 출력
            cv2.imshow("Expert vs Amateur Comparison", combined_frame)
            
            # 종료 조건 (q 키를 누르면 종료)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
    cap_expert.release()
    cap_amateur.release()
    cv2.destroyAllWindows()

# 전문가 영상과 일반인 영상 경로 설정
expert_video_path = "videos1/expert_dance1.mp4"
amateur_video_path = "videos1/amateur_dance1.mp4"

# 함수 실행
process_and_compare_videos(expert_video_path, amateur_video_path, target_fps=30)
