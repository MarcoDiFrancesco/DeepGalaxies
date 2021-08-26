from pathlib import Path

import imageio
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import shutil
import random
import os


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
        assert ds_type in ["train", "validation", "test"]
        self.ds_type = ds_type
        self._generate_dataset()
        self.ds_dir = Path("dataset_generated") / ds_type
        if ds_type == "train" or ds_type == "validation":
            self.images = self._get_imgs_train()
        else:
            self.images = self._get_imgs_test()
        print(f"Dataset {ds_type} of {len(self.images)} images loaded")

    def _generate_dataset(self):
        """Takes images from dataset directory and splits the images in
        dataset_generated/train and dataset_generated/validation
        """
        ds_from = Path("dataset")
        ds_to = Path("dataset_generated")
        if ds_to.exists():
            return
        ds_to.mkdir()
        shutil.copytree(ds_from / "test", ds_to / "test")
        os.mkdir(ds_to / "train")
        os.mkdir(ds_to / "validation")
        for galaxy in self.galaxy_names.values():
            galaxy_dir = ds_from / "train" / galaxy
            os.makedirs(ds_to / "train" / galaxy, exist_ok=True)
            os.makedirs(ds_to / "validation" / galaxy, exist_ok=True)
            for pic in galaxy_dir.iterdir():
                shutil.copy2(
                    pic,
                    ds_to / "train" / galaxy / pic.name
                    if random.random() < 0.8
                    else ds_to / "validation" / galaxy / pic.name,
                )

    def _get_imgs_train(self):
        """List all image from the dataset train and validation directory

        Returns:
            list: tuples with path to image and label
        """
        images = []
        # [Merging, Barred Spiral...]
        for dir in self.ds_dir.iterdir():
            label = 0
            for key, value in self.galaxy_names.items():
                if value == dir.stem:
                    label = key
            # [1.png, 2.png...]
            # for f in list(dir.iterdir())[:50]:
            for f in dir.iterdir():
                images.append((f, label))
        return images

    def _get_imgs_test(self):
        """List all image from the dataset test directory

        Returns:
            list: tuples with path to image and image number
        """

        images = []
        # [0.png, 1.png...]
        for f in self.ds_dir.iterdir():
            # Filename is the image number
            idx = f.stem
            images.append((f, idx))
        return images

    def __getitem__(self, index):
        """Gets picture and apply augmentation if in train epoch

        Args:
            index (int): image index in the dataloader

        Returns:
            tuple: picture array and label (image number for test)
        """

        # For test idx is galaxy (from 0 to 9)
        f, idx = self.images[index]
        array = imageio.imread(f)
        if self.ds_type == "train":
            augmentation = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize(
                    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    # ),
                    transforms.RandomRotation(degrees=(0, 180)),
                    transforms.ColorJitter(brightness=0.5, hue=0.3),
                    transforms.RandomAutocontrast(),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
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
