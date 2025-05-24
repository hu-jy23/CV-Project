#frame_extraction.py
import argparse
import os
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file (e.g., output.mp4)")
    parser.add_argument("--output", type=str, required=True, help="Directory to save extracted frames")
    parser.add_argument("--fps", type=float, default=None, help="Optional: target fps for frame sampling")
    return parser.parse_args()

def extract_frames_from_video(input_path, output_dir, fps=None):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)

    target_fps = fps if fps is not None else orig_fps
    frame_interval = int(round(orig_fps / target_fps))

    idx = 0
    saved_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            out_path = os.path.join(output_dir, f"frame_{saved_idx:04d}.png")
            cv2.imwrite(out_path, frame)
            saved_idx += 1
        idx += 1
    cap.release()

def main():
    args = parse_args()
    video_path = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps // args.fps) if args.fps else 1

    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            filename = os.path.join(output_dir, f"frame_{saved:04d}.png")
            cv2.imwrite(filename, frame)
            saved += 1
        count += 1

    cap.release()
    print(f"Extracted {saved} frames to {output_dir}")

if __name__ == "__main__":
    main()

