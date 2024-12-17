import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import subprocess
import threading
import time

# Mediapipe Pose 초기화
mp_pose = mp.solutions.pose

# 포즈 정규화 함수
def normalize_keypoints(keypoints):
    pelvis = keypoints[24]
    return np.nan_to_num(keypoints - pelvis)

# 유사도 계산 함수
def calculate_similarity(expert_keypoints, user_keypoints):
    expert_flat = expert_keypoints.flatten()
    user_flat = user_keypoints.flatten()
    distance, _ = fastdtw(expert_flat.reshape(-1, 1), user_flat.reshape(-1, 1), dist=euclidean)
    max_distance = 40
    return max(100 - (distance / max_distance) * 100, 0)

# FFmpeg를 사용한 오디오 재생
def play_audio_with_ffmpeg(audio_file, stop_event):
    process = subprocess.Popen(["ffplay", "-nodisp", "-autoexit", audio_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    stop_event.wait()
    process.terminate()

# 전문가 영상에서 키포인트 추출
def extract_keypoints_from_video(video_path):
    keypoints_list = []
    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose() as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                keypoints = np.array([[lmk.x, lmk.y, lmk.z] for lmk in result.pose_landmarks.landmark])
                keypoints_list.append(normalize_keypoints(keypoints))
    cap.release()
    return keypoints_list

# 실시간 웹캠과 영상 비교 함수
def process_webcam_with_overlay_and_compare(expert_keypoints_list, overlay_video_path, audio_path):
    cap_webcam = cv2.VideoCapture(0)
    cap_overlay = cv2.VideoCapture(overlay_video_path)

    fps = cap_overlay.get(cv2.CAP_PROP_FPS) or 30
    frame_duration = 1 / fps
    window_width, window_height = 720, 1080

    stop_event = threading.Event()
    audio_thread = threading.Thread(target=play_audio_with_ffmpeg, args=(audio_path, stop_event))
    audio_thread.start()

    start_time = time.time()
    overlay_started = False  # bigoutput.mp4 시작 여부
    score = 0
    feedback = "Waiting for Pose Detection..."

    with mp_pose.Pose() as pose:
        while cap_webcam.isOpened():
            elapsed_time = time.time() - start_time

            # 웹캠 프레임 읽기
            ret_webcam, frame_webcam = cap_webcam.read()
            if not ret_webcam:
                break

            # bigoutput.mp4 시작 여부 결정 (1초 지연)
            if elapsed_time >= 1.0 and not overlay_started:
                overlay_start_time = time.time()
                overlay_started = True

            # bigoutput.mp4 프레임 읽기 (1초 이후부터 시작)
            if overlay_started:
                ret_overlay, frame_overlay = cap_overlay.read()
                if not ret_overlay:  # 영상 끝나면 다시 시작
                    cap_overlay.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_overlay, frame_overlay = cap_overlay.read()
            else:
                frame_overlay = np.zeros((window_height, window_width, 3), dtype=np.uint8)  # 빈 화면

            # 화면 리사이즈 및 좌우 반전
            frame_webcam = cv2.flip(frame_webcam, 1)
            frame_webcam_resized = cv2.resize(frame_webcam, (window_width, window_height))
            frame_overlay_resized = cv2.resize(frame_overlay, (window_width, window_height))

            # Mediapipe 포즈 감지
            result_webcam = pose.process(cv2.cvtColor(frame_webcam_resized, cv2.COLOR_BGR2RGB))
            if result_webcam.pose_landmarks:
                user_keypoints = np.array([[lmk.x, lmk.y, lmk.z] for lmk in result_webcam.pose_landmarks.landmark])
                normalized_user_keypoints = normalize_keypoints(user_keypoints)
                expert_keypoints = expert_keypoints_list[len(expert_keypoints_list) % len(expert_keypoints_list)]
                score = calculate_similarity(expert_keypoints, normalized_user_keypoints)
                feedback = "Perfect!" if score >= 98 else "Good!" if score >= 93 else "Normal!" if score >= 80 else "Worst..."

            # 결과 출력
            combined_frame = cv2.addWeighted(frame_webcam_resized, 0.3, frame_overlay_resized, 0.7, 0)
            cv2.putText(combined_frame, f"Score: {score:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined_frame, feedback, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Pose Comparison", combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    cap_webcam.release()
    cap_overlay.release()
    cv2.destroyAllWindows()
    stop_event.set()
    audio_thread.join()

# 실행
expert_video_path = "mantra.mp4"
overlay_video_path = "bigoutput.mp4"

print("Extracting expert keypoints...")
expert_keypoints_list = extract_keypoints_from_video(expert_video_path)
process_webcam_with_overlay_and_compare(expert_keypoints_list, overlay_video_path, expert_video_path)
