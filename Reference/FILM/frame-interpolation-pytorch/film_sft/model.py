# model.py

import torch
import torch.nn as nn
from interpolator import Interpolator

def load_model_from_pth(pth_path: str, device="cuda" if torch.cuda.is_available() else "cpu") -> nn.Module:
    model = Interpolator()
    state_dict = torch.load(pth_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
