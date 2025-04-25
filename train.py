import torch
import torch.optim as optim
import os
import wandb
from torch.utils.data import DataLoader
from models.unet_colorizer import UNetColorizer
from losses.perceptual_loss import PerceptualLoss
from data.dataloader import VideoDataset
from utils.visualization import visualize_results
from utils.metrics import psnr, ssim
import yaml
from torch.utils.tensorboard import SummaryWriter

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup WandB
wandb.init(project=config['logging']['wandb_project_name'], config=config)

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetColorizer().to(device)
optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'], weight_decay=config['train']['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['num_epochs'])
criterion = PerceptualLoss()

# Setup DataLoader
train_dataset = VideoDataset(config['data']['dataset_path'], config['data']['frame_size'])
train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)

# Setup TensorBoard
writer = SummaryWriter(config['logging']['tensorboard_log_dir'])

# Training loop
for epoch in range(config['train']['num_epochs']):
    model.train()
    total_loss = 0
    for i, (frames, _) in enumerate(train_loader):
        frames = frames.to(device)  

        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(frames)
        
        # Compute loss
        loss = criterion(outputs, frames)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        # Logging to TensorBoard and WandB
        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)
        wandb.log({'epoch': epoch, 'loss': loss.item()})

    scheduler.step()
    print(f"Epoch [{epoch+1}/{config['train']['num_epochs']}], Loss: {total_loss/len(train_loader)}")

    # Save model checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), os.path.join('checkpoints', f'colorizer_epoch{epoch+1}.pth'))

writer.close()
