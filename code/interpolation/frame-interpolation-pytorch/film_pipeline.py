import os
import argparse
import cv2
import torch
import glob

from interpolator import Interpolator
from util import read_image, save_image
from frame_extraction import extract_frames_from_video  # You already have this
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input video (mp4)')
    parser.add_argument('--model', type=str, required=True, help='Path to FILM .pt model')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save output frames')
    parser.add_argument('--cleanup', action='store_true', help='Delete temp_frames after processing')
    return parser.parse_args()

@torch.no_grad()
def interpolate_all_pairs(model, input_folder, output_folder, device='cuda'):
    images = sorted(glob.glob(os.path.join(input_folder, "*.png")))

    os.makedirs(output_folder, exist_ok=True)
    idx = 0

    for i in tqdm(range(len(images) - 1), desc="Interpolating"):
        img0, _ = read_image(images[i])
        img0 = img0.to(device)
        img1, _ = read_image(images[i + 1])
        img1 = img1.to(device)
        dt = torch.tensor([[0.5]], dtype=torch.float32).to(device)
        interpolated = model(img0, img1, dt).clamp(0, 1)

        # Save: original img0, interpolated, then img1 (next loop skips writing img1 again)
        save_image(interpolated[0].cpu(), os.path.join(output_folder, f"frame_{idx+1:04d}.png"))
        if i == 0:
            save_image(img0.cpu(), os.path.join(output_folder, f"frame_{idx:04d}.png"))
        save_image(img1.cpu(), os.path.join(output_folder, f"frame_{idx+2:04d}.png"))
        idx += 2

def main():
    args = parse_args()
    temp_dir = "./temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    print("📽️ Extracting frames...")
    extract_frames_from_video(args.input, temp_dir)

    print("📦 Loading model...")
    #model = torch.jit.load(args.model).eval().to("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = torch.jit.load(args.model).eval().to(device)

    print("🎬 Interpolating...")
    interpolate_all_pairs(model, temp_dir, args.save_dir, device=device)

    if args.cleanup:
        import shutil
        shutil.rmtree(temp_dir)
        print("🧹 Temp frames cleaned.")

    print(f"✅ Done! All frames saved to {args.save_dir}")

if __name__ == '__main__':
    main()