def main():
    overlay_video = "contourwithaudio.mp4"
    audio_file = "contourwithaudio.mp4"
    expert_keypoints_file = "expert_keypoints.npy"

    stop_event = threading.Event()

    print("Loading expert keypoints...")
    expert_keypoints = load_keypoints_from_file(expert_keypoints_file)

    try:
        webcam_with_overlay_and_comparison(audio_file, overlay_video, expert_keypoints, stop_event)
    finally:
        stop_event.set()
        print("Program Finished.")


if __name__ == "__main__":
    mp_pose = mp.solutions.pose
    main()
