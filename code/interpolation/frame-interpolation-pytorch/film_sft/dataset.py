# dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class FilmTripletDataset(Dataset):
    def __init__(self, root_dir, image_size=(512, 512)):
        self.root_dir = root_dir
        self.scene_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        scene = self.scene_dirs[idx]
        to_gray_rgb = lambda p: Image.open(os.path.join(scene, p)).convert('L').convert('RGB')

        x0 = to_gray_rgb('frame_0.png')
        xt = to_gray_rgb('frame_t.png')
        x1 = to_gray_rgb('frame_1.png')

        return {
            'x0': self.transform(x0),
            'xt': self.transform(xt),
            'x1': self.transform(x1),
            'dt': 0.5
        }
