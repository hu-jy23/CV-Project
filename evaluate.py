import torch
import os
from models.unet_colorizer import UNetColorizer
from utils.metrics import psnr, ssim
import yaml
from data.dataloader import VideoDataset

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetColorizer().to(device)
model.load_state_dict(torch.load(os.path.join('checkpoints', 'colorizer_epoch100.pth')))
model.eval()

# Evaluate
test_dataset = VideoDataset(config['data']['dataset_path'], config['data']['frame_size'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

total_psnr = 0
total_ssim = 0

for i, (frames, _) in enumerate(test_loader):
    frames = frames.to(device)
    with torch.no_grad():
        output = model(frames)
        
    psnr_value = psnr(output, frames)
    ssim_value = ssim(output, frames)
    
    total_psnr += psnr_value
    total_ssim += ssim_value

print(f"Average PSNR: {total_psnr / len(test_loader)}")
print(f"Average SSIM: {total_ssim / len(test_loader)}")
