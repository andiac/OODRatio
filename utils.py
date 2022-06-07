import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class OE_Dataset(Dataset):
    def __init__(self, npy_file, transform=None):
        self.d = np.load(npy_file)
        self.transform = transform

    def __len__(self):
        return self.d.shape[0]

    def __getitem__(self, idx):
        return self.transform(self.d[idx])

class RandomSizeCrop(torch.nn.Module):
    def __init__(self, sizes, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()
        self.randomcrops = []
        for size in sizes:
            self.randomcrops.append(transforms.RandomCrop(size, padding, pad_if_needed, fill, padding_mode))
        
    def forward(self, img):
        l = len(self.randomcrops)
        idx = random.randrange(l)
        return self.randomcrops[idx](img)

