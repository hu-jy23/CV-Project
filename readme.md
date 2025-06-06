# Black-and-White Video Restoration via Frame Interpolation and Colorization 

##  Overview

### This project restores old grayscale (black-and-white) videos by combining two major components:

1.  **Frame Interpolation** — Fine-tuned FILM model trained on grayscale triplets.
2.  **Video Colorization** — Modified TCVC with multi-reference temporal consistency.

The system enhances **continuity**, **color richness**, and **temporal smoothness** of vintage footage.

The following process runs successfully on ``Windows`` systems.

---

## 📁 Project Structure

```text
cvproj_submission/
├── code/
│   ├── colorization/              # TCVC model and inference code
│   ├── interpolation/             # FILM model and grayscale SFT
│   ├── utils/                     # Preprocessing and helpers
│   ├── test/                      # Example inputs
│   └── requirements.txt
├── figures/                       # Result images and comparison plots
├── videos/                        # Demo videos (optional)
└── README.md
```
## 📽️ Demo videos are provided under `/demos`.
```
demos:
    - origin.png & sft.png shows the effect of FILM-SFT model
    - medium.mp4 and medium_inter.mp4 shows the effect of interpolation on early cases.
    - med_color.mp4 shows colorization on medium.mp4 and corresponding frames stored in frames_med
    - medium_inter_color.mp4 shows colorization on medium_inter.mp4
    - difficult1.mp4 and difficult2.mp4 shows the final effect of the most difficult cases in the test with GT as ref frames to garantee the best results in output. Which shows its power in semi-supervised Task.
      (We can recoganize that some illumination on human face & cloth is still hard to deal with)
    - input.mp4 is the groundtruth  of the test video
    - ref_frames collects our phased outputs from cross-ref TCVC model
```
---

## 🛠️ Environment Setup Instructions

```bash
conda create -n bw_restore python=3.9
conda activate bw_restore
pip install -r requirements.txt
```

---

# 🔧 Usage

## ▶ Interpolation Using Exported Model
### QuickStart
First, download the checkpoint of our FILM-SFT model or the original FILM model from [Google Drive](https://drive.google.com/drive/folders/1XwXQZw_q5_Z-5_Yyz7yjw_jXQ5Q_YxQ5?usp=sharing).

The following script creates an MP4 video of interpolated frames between two input images:
```bash
cd code/interpolation/frame-interpolation-pytorch
python inference.py "model_path" "img1" "img2" [--save_path SAVE_PATH] [--gpu] [--fp16] [--frames FRAMES] [--fps FPS]
```
* `model_path`: Path to the exported TorchScript checkpoint
* `img1`: Path to the first image
* `img2`: Path to the second image
* `--save_path SAVE_PATH`: Path to save the interpolated video. If not provided, it defaults to the location of `img1` as `output.mp4`
* `--gpu`: Use GPU if available
* `--fp16`: Use float16 for faster inference on compatible GPUs
* `--frames`: Number of intermediate frames to generate
* `--fps`: FPS of the output video

### Apply on a Video
Note: This may take longer since optimization is limited under TorchScript.
```bash
cd code/interpolation/
python film_pipeline.py --input input.mp4 --model model\model_name.pt --save_dir output/set --cleanup
```
> Make sure to create the `output/` folder before running the script.

## ▶ Frame Extraction from Interpolated Video

```bash
python frame_extraction_cli.py --input ./output/set/output.mp4 --output ./output/set/frames --fps 15
```

* `--input`: Path to the interpolated `.mp4` video file
* `--output`: Directory where extracted frames will be saved
* `--fps`: (Optional) Sampling FPS. If not specified, all frames will be extracted

---

## ▶ Colorization via TCVC

Open terminal in `code/colorization/TCVC` folder.

### 🔄 One-Step Pipeline
```bash
python run_pipeline.py
```
Ensure that `input.mp4` is placed under the `./test/` directory. Modify paths in `run_pipeline.py` if needed.

### ▶ Step-by-Step Execution:

### Step 1: Convert video to frames
```
python video_to_frames.py ./test/input.mp4 --color "./test/color" --bw "./test/input_frames"
```
### Step 2: Extract reference frame
```
python extract_ref_frame.py --input "./test/input_frames" --output "./test/stage1_input/stage1_input" --all_frame_output  "./test/stage2_input"
```
### Use ground-truth reference frames (optional)
```
python extract_ref_frame.py --input "./test/input_frames" --output "./test/stage1_input/stage1_input" --all_frame_output "./test/stage2_input" --use_ground_truth_ref --gt_source "./test/color" --gt_target "./test/stage1_output"
```
### Step 3: Stage 1 - Reference frame colorization
```
python stage1/test.py --data_root_val "./test/stage1_input" --test_video_output_path "./test/stage1_output"
```
### Step 4: Stage 2 - Full sequence colorization
```
python stage2/inference_colorvid.py --test_path "./test/stage2_input" --ref_path "./test/stage1_output" --test_output_path "./test/stage2_output"
```
### Step 5: Convert colorized frames back to video
```
python image_sequence_to_video.py --input ./test/stage2_output --output ./test/final_output --fps 30
```
> Note: You may need to manually adjust `test_video_size` and `img_size` in Stage 1 and Stage 2 if the video resolution does not match.

---

## 📊 Results

| Stage          | Input                     | Output                       |
|----------------|----------------------------|-------------------------------|
| Interpolation  | Grayscale frames           | Smooth high-FPS video         |
| Colorization   | Grayscale + Ref images     | Colored frames                |

See `/figures` and `/videos` for visual examples.

---

## 📚 Dataset Used

- [Vimeo-90K Triplet](http://toflow.csail.mit.edu/)
- Grayscale test clips (custom)

---

## 👥 Authors

- Jiayi Hu (Tsinghua University, Yao Class)
- Xuanyi Xie (Tsinghua University, Yao Class)

---

## 📌 Notes

- See `Final_Report.pdf` for detailed explanation of models and experiments.
## 📌 Github repo: https://github.com/hu-jy23/CV-Project/tree/final_submission