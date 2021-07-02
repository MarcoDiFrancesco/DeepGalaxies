from pathlib import Path

import imageio
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset
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
        self.images = (
            self._get_imgs_train() if ds_type == "train" else self._get_imgs_test()
        )
        print(f"Dataset {ds_type} of {len(self.images)} images loaded")

    def _get_imgs_train(self):
        images = []
        # [Merging, Barred Spiral...]
        for dir in self.ds_dir.iterdir():
            label = 0
            for key, value in self.galaxy_names.items():
                if value == dir.stem:
                    label = key
            # [1.png, 2.png...]
            for f in dir.iterdir():
                images.append((f, label))
        return images

    def _get_imgs_test(self):
        images = []
        # [0.png, 1.png...]
        for f in self.ds_dir.iterdir():
            # Filename is the image number
            idx = f.stem
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
        else:
            augmentation = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        array = augmentation(array)
        return array, idx

    def __len__(self):
        return len(self.images)
