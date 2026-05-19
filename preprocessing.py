import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import glob
from tqdm import tqdm
import gc

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

INPUT_DIR = "/Users/aryankrishnan/.cache/huggingface/hub/datasets--akasheroor--American-Sign-Language-Dataset/snapshots/e7979505c0dff7072ef36d45b3cddfffb50ba871"
OUTPUT_DIR = "mediapipe_landmarks"

HAND_MODEL = "hand_landmarker.task"

TARGET_FPS = 12
TARGET_W = 640
TARGET_H = 480

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

GPU = BaseOptions.Delegate.GPU
CPU = BaseOptions.Delegate.CPU


def make_options(delegate=GPU):
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL, delegate=delegate),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return hand_options


def normalize_hand(lm_list):
    pts = np.array([[p.x, p.y, p.z] for p in lm_list], dtype=np.float32)

    # wrist-centered
    pts = pts - pts[0]

    # remove wrist landmark
    pts = pts[1:]

    # normalize scale
    scale = np.max(np.linalg.norm(pts, axis=1))
    pts /= scale + 1e-6

    return pts


def extract_hands(result):
    hands = np.zeros((2, 20, 3), dtype=np.float32)

    for i, handedness in enumerate(result.handedness):
        if i >= 2:
            break

        slot = 0 if handedness[0].category_name == "Left" else 1

        hands[slot] = normalize_hand(result.hand_landmarks[i])

    return hands


def process_video(video_path, output_dir, hand_lm, global_ts_ms):
    video_filename = os.path.basename(video_path)
    video_name_without_ext = os.path.splitext(video_filename)[0]
    parent_folder_name = os.path.basename(os.path.dirname(video_path))

    output_filename = f"{parent_folder_name}_{video_name_without_ext}.npz"
    out_path = os.path.join(output_dir, output_filename)

    if os.path.exists(out_path):
        return f"SKIP {output_filename}", global_ts_ms

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return f"FAIL {output_filename}", global_ts_ms

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_interval = max(1, round(src_fps / TARGET_FPS))

    hands_out = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_resized = cv2.resize(frame, (TARGET_W, TARGET_H))

            frame_rgba = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGBA)
            frame_rgba = np.ascontiguousarray(frame_rgba)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=frame_rgba)

            timestamp_ms = global_ts_ms + int(frame_idx / src_fps * 1000)

            result = hand_lm.detect_for_video(mp_image, timestamp_ms)

            hands_out.append(extract_hands(result))

            del mp_image
            del result

            if frame_idx % 100 == 0:
                gc.collect()

        frame_idx += 1

    cap.release()

    if not hands_out:
        return f"EMPTY {output_filename}", global_ts_ms

    np.savez_compressed(
        out_path,
        hands=np.stack(hands_out, axis=0),
    )

    global_ts_ms += int(total_frames / src_fps * 1000) + 1000

    return f"OK {output_filename} → T={len(hands_out)}", global_ts_ms


def _run(video_paths, hand_lm):
    global_ts_ms = 0

    use_tqdm = sys.stdout.isatty()

    iterator = tqdm(video_paths) if use_tqdm else video_paths

    for idx, vp in enumerate(iterator):
        result, global_ts_ms = process_video(vp, OUTPUT_DIR, hand_lm, global_ts_ms)

        if use_tqdm:
            if not (result.startswith("OK") or result.startswith("SKIP")):
                tqdm.write(result)
        else:
            if idx % 10 == 0:
                print(f"[{idx}/{len(video_paths)}] {result}", flush=True)

    print("Done.", flush=True)


def main():
    # ------------------------------------
    # Chunk arguments
    # ------------------------------------
    if len(sys.argv) < 3:
        print("Usage:")
        print("python preprocess.py <start_idx> <end_idx>")
        sys.exit(1)

    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_paths = sorted(
        glob.glob(os.path.join(INPUT_DIR, "**", "*.mp4"), recursive=True)
    )

    total_videos = len(video_paths)

    # Clamp bounds safely
    start_idx = max(0, start_idx)
    end_idx = min(end_idx, total_videos)

    video_chunk = video_paths[start_idx:end_idx]

    print(f"Found {total_videos} total videos")
    print(
        f"Processing chunk: " f"{start_idx}:{end_idx} " f"({len(video_chunk)} videos)"
    )

    if len(video_chunk) == 0:
        print("No videos in selected chunk.")
        return

    try:
        hand_opts = make_options(delegate=GPU)

        with HandLandmarker.create_from_options(hand_opts) as hand_lm:
            print("Running on GPU")
            _run(video_chunk, hand_lm)

    except Exception as e:
        print(f"GPU failed ({e}), falling back to CPU")

        hand_opts = make_options(delegate=CPU)

        with HandLandmarker.create_from_options(hand_opts) as hand_lm:
            print("Running on CPU")
            _run(video_chunk, hand_lm)


if __name__ == "__main__":
    main()
