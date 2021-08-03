from pathlib import Path
from torch.utils.data import DataLoader
from Dataset import MyDataset
from Trainer import Trainer
import os
import csv


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
dataset = MyDataset("test")
dl = DataLoader(dataset, batch_size=32)
model_path = Path("logs") / "2021-07-01 16:05:16" / "best.pth"
trainer = Trainer(model_path)
predictions = trainer.test_epoch(dl)
with open("output.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(predictions)
print("Done!")
