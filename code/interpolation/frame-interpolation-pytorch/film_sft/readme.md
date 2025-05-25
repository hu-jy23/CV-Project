## 假设我们将来的数据目录结构为：
train_data/
├── scene1/
│   ├── frame_0.png
│   ├── frame_t.png  ← GT
│   ├── frame_1.png
├── scene2/
│   └── ...


## 📁 文件/目录准备建议：
film_sft/
├── model.py
├── dataset.py
├── train.py
├── model/
│   └── film_style_state.pth
├── train_data/
│   ├── scene1/
│   │   ├── frame_0.png
│   │   ├── frame_t.png
│   │   ├── frame_1.png
│   └── ...
└── checkpoints/
