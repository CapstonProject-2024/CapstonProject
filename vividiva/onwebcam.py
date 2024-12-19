import cv2
import threading
import time
import subprocess

def play_audio_and_wait(audio_file, speed):
    """
    오디오를 재생하고 준비 시간을 기다림.
    """
    # FFplay 명령어 실행
    process = subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-af", f"atempo={speed}", audio_file],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # 오디오가 준비된 상태로 간주하기 위해 약간 대기
    time.sleep(0.5)  # FFplay가 안정적으로 시작되도록 대기
    return process

def webcam_with_overlay(audio_file, overlay_video, stop_event, speed=1.0):
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
    frame_interval = (1 / fps_overlay) / speed if fps_overlay > 0 else 0.033 / speed

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

    # 오디오 재생 및 준비 시간 대기
    print("Playing audio and preparing overlay.")
    audio_process = play_audio_and_wait(audio_file, speed)

    # 오버레이 출력
    start_time = time.time()
    while not stop_event.is_set():
        elapsed_time = (time.time() - start_time) * speed

        ret_webcam, frame_webcam = cap_webcam.read()
        if not ret_webcam:
            print("Failed to capture webcam frame.")
            stop_event.set()
            break

        cap_overlay.set(cv2.CAP_PROP_POS_MSEC, elapsed_time * 1000)  # 동기화된 시간으로 이동
        ret_overlay, frame_overlay = cap_overlay.read()
        if not ret_overlay:
            print("Restarting overlay video.")
            cap_overlay.set(cv2.CAP_PROP_POS_FRAMES, 0)
            start_time = time.time()
            continue

        if audio_process.poll() is not None:  # 오디오가 종료되었으면
            print("Restarting audio.")
            audio_process.terminate()
            audio_process = play_audio_and_wait(audio_file, speed)

        # 거울 모드로 변경
        frame_webcam = cv2.flip(frame_webcam, 1)

        # 프레임 크기 조정 및 합성
        frame_webcam_resized = cv2.resize(frame_webcam, (width, height))
        frame_overlay_resized = cv2.resize(frame_overlay, (width, height))
        combined_frame = cv2.addWeighted(frame_webcam_resized, 1, frame_overlay_resized, 1, 0)

        # 화면 출력
        cv2.imshow("Dance App", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

        time.sleep(frame_interval)

    cap_webcam.release()
    cap_overlay.release()
    cv2.destroyAllWindows()
    audio_process.terminate()

def main():
    overlay_video = "contour/contourwithaudio_x100.mp4"
    audio_file = "contour/contourwithaudio_x100.mp4"

    stop_event = threading.Event()

    try:
        webcam_with_overlay(audio_file, overlay_video, stop_event)
    finally:
        stop_event.set()
        print("Program Finished.")

if __name__ == "__main__":
    main()
