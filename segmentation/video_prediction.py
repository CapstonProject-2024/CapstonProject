original_video = cv2.VideoCapture("/content/drive/MyDrive/Colab Notebooks/vividiva/vividiva.mp4")
fps = original_video.get(cv2.CAP_PROP_FPS)

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "/content/drive/MyDrive/Colab Notebooks/vividiva/test"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
frame_idx = 0
plt.figure(figsize=(9, 6))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))

inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)


#############################################################
######## Add a second click to refine the prediction ########
#############################################################


# 전신
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 0  # give a unique id to each object we interact with (it can be any integers)

# Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
# sending all clicks (and their labels) to `add_new_points_or_box`
points_0 = np.array([[200, 300], [200, 100]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels_0 = np.array([1, 1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points_0,
    labels=labels_0,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points_0, labels_0, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

# 팔
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Let's add a positive click at (x, y) = (210, 350) to get started
points_1 = np.array([[200, 240]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels_1 = np.array([1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points_1,
    labels=labels_1,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points_1, labels_1, plt.gca())
show_mask((out_mask_logits[1] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[1])

# 하의
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

# Let's add a positive click at (x, y) = (210, 350) to get started
points_2 = np.array([[200, 320], [200, 270], [140, 470], [270, 470]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels_2 = np.array([1, 0, 1, 1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points_2,
    labels=labels_2,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points_2, labels_2, plt.gca())
show_mask((out_mask_logits[2] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[2])



#############################################################
######## run propagation throughout the video and collect the results in a dict ########
#############################################################


video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 30
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)



#############################################################
################ save segmentation video ################
#############################################################

## 동영상 원본 fps
# 동영상 파일 경로
video_path = "/content/drive/MyDrive/Colab Notebooks/vividiva/vividiva.mp4"

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

if cap.isOpened():
    # FPS 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
else:
    print("동영상을 열 수 없습니다.")

# 캡처 객체 해제
cap.release()


## 비디오 저장
# 비디오 저장 경로와 설정
output_video_path = "/content/drive/MyDrive/Colab Notebooks/vividiva/segmentation.mp4"
frame_size = None  # 프레임 크기 (자동 설정)

for frame_idx, frame_name in enumerate(frame_names):
    # 원본 프레임 불러오기
    frame = np.array(Image.open(os.path.join(video_dir, frame_name)))

    # 첫 번째 프레임에서 비디오 크기 설정
    if frame_size is None:
        frame_size = (frame.shape[1], frame.shape[0])  # (width, height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # 세그멘테이션 결과 시각화
    overlay_frame = frame.copy()
    if frame_idx in video_segments:
        for out_obj_id, out_mask in video_segments[frame_idx].items():
            # 배치 차원 제거
            if out_mask.ndim == 3 and out_mask.shape[0] == 1:
                out_mask = np.squeeze(out_mask, axis=0)  # (1, H, W) → (H, W)

            # 마스크를 컬러로 변환
            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            mask_indices = out_mask > 0
            colored_mask[mask_indices] = [255, 255, 255]  # 흰색 (RGB)

            # 투명도 설정
            alpha = 0.5
            overlay_frame = cv2.addWeighted(overlay_frame, 1 - alpha, colored_mask, alpha, 0)

    # 비디오에 프레임 추가
    video_writer.write(cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2BGR))

# 비디오 작성기 종료
video_writer.release()
print(f"비디오 저장 완료: {output_video_path}")
