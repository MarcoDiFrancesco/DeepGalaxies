from pathlib import Path
from torch.utils.data import DataLoader
from Dataset import MyDataset
from Trainer import Trainer
import os
import csv


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dataset = MyDataset("test")
dl = DataLoader(dataset, batch_size=512)
model_path = (
    Path("/")
    / "thunderdisk"
    / "data_rene_policistico_log"
    / "2021-08-23 11:51:08"
    / "535.pth"
)
trainer = Trainer(model_path)
predictions = trainer.test_epoch(dl)
with open("output.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(predictions)
print("Done!")
