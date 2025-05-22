# train.py

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from losses import build_loss
from model import load_model_from_pth
from dataset import FilmTripletDataset

@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    for batch in val_loader:
        x0 = batch['x0'].to(device)
        xt = batch['xt'].to(device)
        x1 = batch['x1'].to(device)
        dt = torch.tensor([[batch['dt']]] * len(x0), dtype=torch.float32, device=device)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            pred = model(x0, x1, dt)
            loss = criterion(pred, xt)
        val_loss += loss.item()
    model.train()
    return val_loss / len(val_loader)


# ==== 创建检查点目录 ====
os.makedirs("checkpoints", exist_ok=True)

# ==== 配置参数 ====
USE_AMP = True

DATA_ROOT = "train_data_split"                     # 替换为你的数据集目录
PRETRAINED_PTH = "model/film_style_state.pth"
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4
IMAGE_SIZE = (256, 256)
USE_AMP = True



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Dataset ==== ✅ 使用 train/val 两套数据
train_set = FilmTripletDataset(os.path.join(DATA_ROOT, "train"), image_size=IMAGE_SIZE)
val_set   = FilmTripletDataset(os.path.join(DATA_ROOT, "val"), image_size=IMAGE_SIZE)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

# ==== Model ====
model = load_model_from_pth(PRETRAINED_PTH, device=device)
model.train()

# ==== Loss + Optimizer ====
LOSS_TYPE = "l1"  # 也可设为 "charbonnier+lpips"               LOSS_TYPE = "charbonnier+lpips"
criterion = build_loss(LOSS_TYPE).to(device)


optimizer = optim.AdamW(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler() if USE_AMP else None


wandb.init(
    project="film_sft",       # 你可以自定义项目名
    name="sft_l1_ssim",       # 每次运行的名称（建议带 loss 类型）
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "image_size": IMAGE_SIZE,
        "loss_type": LOSS_TYPE,
        "amp": USE_AMP,
    }
)
print(f"Trainset size: {len(train_loader.dataset)} samples")
global_step = 0  # 在训练循环外定义
# ==== 训练循环 ====
for epoch in range(EPOCHS):
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in pbar:
        print(".", end="", flush=True)  # 每 batch 输出一个点
        x0 = batch['x0'].to(device)
        xt = batch['xt'].to(device)
        x1 = batch['x1'].to(device)
        dt = torch.full((x0.size(0), 1), 0.5, dtype=torch.float32, device=device)


        optimizer.zero_grad()
        if USE_AMP:
            with torch.cuda.amp.autocast():
                pred = model(x0, x1, dt)
                loss = criterion(pred, xt)
                if torch.isnan(loss):
                    print(f"⚠️ Loss is NaN at step {global_step}, skipping this batch.")
                    continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x0, x1, dt)
            loss = criterion(pred, xt)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        global_step += 1
        pbar.set_postfix(loss=running_loss / (pbar.n + 1))
        wandb.log({"train/loss_step": loss.item(), "global_step": global_step})

    avg_train_loss = running_loss / len(train_loader)
    val_loss = evaluate(model, val_loader, criterion, device)  # ✅ 验证集 loss

    wandb.log({
        "epoch": epoch + 1,
        "train/loss_epoch": avg_train_loss,
        "val/loss": val_loss,
        "global_step": global_step
    })
    if (epoch + 1) % 3 == 0:
        # 每3个epoch保存一次模型
        torch.save(model.state_dict(), f"checkpoints/film_epoch{epoch+1}.pth")

    torch.save(model.state_dict(), f"checkpoints/film_epoch_last.pth")

print("✅ Training finished.")
