import torch
from pathlib import Path
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset
import numpy as np


class MyDataset(Dataset):
    def __init__(self, ds_type):
        self.ds_type = ds_type
        self.ds_dir = Path("dataset") / ds_type
        self.images = [f for f in self.ds_dir.iterdir()]
        print(f"Dataset of {len(self.images)} images loaded")

    def get_split(self, train_len):
        train_len = int(train_len * len(self))
        valid_len = len(self) - train_len
        return random_split(self, [train_len, valid_len])

    def __getitem__(self, index):
        f = self.images[index]
        img = np.load(f)
        # Only for train augment
        if self.ds_type == "train":
            pass
            # preprocess = transforms.Compose([
            #    transforms.Resize(256),
            #    transforms.CenterCrop(224),
            #    transforms.ToTensor(),
            #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ])
            # img = preprocess(img)

        # For train/validation idx is the id (e.g. 3678)
        # For test idx is galaxy (from 0 to 9)
        idx = int(f_name.stem)
        print("TODO: is it right???", idx)
        return (img, idx)

    def __len__(self):
        return len(self.images)
