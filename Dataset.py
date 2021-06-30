from pathlib import Path
import torch
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset
import numpy as np
import imageio
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, ds_type):
        self.galaxy_names = {
            0: "Edge-on without Bulge",
            1: "Unbarred Tight Spiral",
            2: "Edge-on with Bulge",
            3: "Merging",
            4: "In-between Round Smooth",
            5: "Barred Spiral",
            6: "Disturbed",
            7: "Unbarred Loose Spiral",
            8: "Cigar Shaped Smooth",
            9: "Round Smooth",
        }
        assert ds_type in ["train", "test"]
        self.ds_type = ds_type
        self.ds_dir = Path("dataset") / ds_type
        self.images = self.get_images()  # [:50]

        print(f"Dataset of {len(self.images)} images loaded")

    def get_images(self):
        images = []
        # {Merging, Barred Spiral...}
        for dir in self.ds_dir.iterdir():
            idx = 0
            for key, value in self.galaxy_names.items():
                if value == dir.stem:
                    idx = key
            # [1.png, 2.png...]
            for f in dir.iterdir():
                images.append((f, idx))
        return images

    def get_split(self, train_len):
        train_len = int(train_len * len(self))
        valid_len = len(self) - train_len
        return random_split(self, [train_len, valid_len])

    def __getitem__(self, index):
        # For test idx is galaxy (from 0 to 9)
        f, idx = self.images[index]
        array = imageio.imread(f)
        if self.ds_type == "train":
            # transforms = transforms.Compose(
            #     [
            #         transforms.Resize(256),
            #         transforms.CenterCrop(224),
            #         transforms.ToTensor(),
            #         transforms.Normalize(
            #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            #         ),
            #     ]
            # )
            augmentation = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    transforms.RandomRotation(degrees=(0, 180)),
                    transforms.ColorJitter(brightness=0.5, hue=0.3),
                ]
            )
            array = augmentation(array)
        # TODO: test if this was needed
        # array = np.transpose(array, (2, 0, 1))

        # array = torch.Tensor(array)
        return array, idx

    def __len__(self):
        return len(self.images)
