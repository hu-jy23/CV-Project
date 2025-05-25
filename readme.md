# Black-and-White Video Restoration via Frame Interpolation and Colorization 

##  Overview

### This project restores old grayscale (black-and-white) videos by combining two major components:

1.  **Frame Interpolation** — Fine-tuned FILM model trained on grayscale triplets.
2.  **Video Colorization** — Modified TCVC with multi-reference temporal consistency.

The system enhances **continuity** **color richness** and **temporal smoothness** of vintage footage.

The following process can run successfully on ``Windows``
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

---

## 🛠️ Environment Setup Instructions

```bash
conda create -n bw_restore python=3.9
conda activate bw_restore
pip install -r requirements.txt
```

---

# 🔧 Usage

## ▶ Interpolation Using exported model
### QuickStart
First, you need to download the checkpoint of our FILM-sft model or the original FILM model on [Quark CloudDisk](https://drive.google.com/drive/folders/1XwXQZw_q5_Z-5_Yyz7yjw_jXQ5Q_YxQ5?usp=sharing).

The following script creates an MP4 video of interpolated frames between 2 input images:
```
cd code/interpolation/frame-interpolation-pytorch
python inference.py "model_path" "img1" "img2" [--save_path SAVE_PATH] [--gpu] [--fp16] [--frames FRAMES] [--fps FPS]
```
* ```model_path``` Path to the exported TorchScript checkpoint
* ```img1``` Path to the first image
* ```img2``` Path to the second image
* ```--save_path SAVE_PATH``` Path to save the interpolated frames as a video, if absent it will be saved in the same directory as ```img1``` is located and named ```output.mp4```
* ```--gpu``` Whether to attempt to use GPU for predictions
* ```--fp16``` Whether to use fp16 for calculations, speeds inference up on GPUs with tensor cores
* ```--frames FRAMES``` Number of frames to interpolate between the input images
* ```--fps FPS``` FPS of the output video
---

### Apply on your video
This probably need a long time due to uncapable to optimaze on torchscript.
```
cd code/interpolation/
python film_pipeline.py --input input.mp4 --model model\model_name.pt --save_dir output/set --cleanup
```
Notice that you should create the folder "output/" before application.

## ▶ Colorization

```bash
python colorization/infer.py \
  --input test/gray_video.mp4 \
  --refs test/ref_images/ \
  --output results/colored/
```

## 📊 Results

| Stage          | Input                     | Output                       |
|----------------|----------------------------|-------------------------------|
| Interpolation  | Grayscale frames             | Smooth high-FPS video         |
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
