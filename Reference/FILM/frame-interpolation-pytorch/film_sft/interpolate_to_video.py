import os
import torch
from PIL import Image
import torchvision.transforms as T
import imageio
from tqdm import tqdm

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("model/film_style.pt", map_location=device).eval()

# 图像预处理
transform = T.Compose([
    T.Resize((512, 512)),   # 按需调整
    T.ToTensor()
])

# 加载输入帧
frame_dir = "frames/"  # 文件夹中为 frame_000.png, frame_001.png, ...
frames = sorted([
    os.path.join(frame_dir, f)
    for f in os.listdir(frame_dir)
    if f.endswith(".png") or f.endswith(".jpg")
])

def load_image(path):
    return transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

# 帧插值函数
def interpolate_pair(img0, img1):
    dt = torch.tensor([[0.5]], dtype=torch.float32).to(device)
    with torch.no_grad():
        result = model(img0, img1, dt)
    return result

# 生成帧序列：原图 + 插帧
output_frames = []

# 帧插值函数，插入 n 帧
def interpolate_frames(img0, img1, num_interp):
    outputs = []
    for i in range(1, num_interp + 1):
        dt = torch.tensor([[i / (num_interp + 1)]], dtype=torch.float32).to(device)
        with torch.no_grad():
            mid = model(img0, img1, dt)
        outputs.append(mid)
    return outputs

# 清空输出帧
output_frames = []

# 插帧主循环
for i in tqdm(range(len(frames) - 1)):
    img0 = load_image(frames[i])
    img1 = load_image(frames[i + 1])

    mids = interpolate_frames(img0, img1, num_interp=60)  # 这里设置插帧数量

    # 解码回 PIL
    def to_pil(img_tensor):
        img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
        return Image.fromarray((img * 255).astype('uint8'))

    output_frames.append(to_pil(img0))      # 原始第一帧
    output_frames.extend([to_pil(m) for m in mids])  # 插入帧们

output_frames.append(to_pil(load_image(frames[-1])))  # 原始最后一帧


# 保存为 GIF
gif_path = "output.gif"
output_frames[0].save(
    gif_path, save_all=True, append_images=output_frames[1:], duration=100, loop=0
)

# 保存为 MP4（使用 imageio）
mp4_path = "output.mp4"
imageio.mimsave(mp4_path, [f for f in output_frames], fps=10)

print(f"✅ GIF saved to {gif_path}")
print(f"✅ MP4 saved to {mp4_path}")
