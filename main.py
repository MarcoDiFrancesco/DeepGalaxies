from Trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import os
import csv


def print_accuracy():
    print("--- Train dataset ---")
    acc, loss, accuracy_by_label = trainer.valid_epoch(trainer.train_dl)
    _print_stats(acc, loss, accuracy_by_label)
    print("--- Valid dataset ---")
    acc, loss, accuracy_by_label = trainer.valid_epoch(trainer.valid_dl)
    _print_stats(acc, loss, accuracy_by_label)


# Print train and validation perfornces by galaxy
def _print_stats(acc, loss, accuracy_by_label):
    print(f"Accuracy {acc:.2f}%    ")
    print(f"Loss {loss:8f}")
    for key, value in accuracy_by_label.items():
        galaxy_name = trainer.dataset.galaxy_names[key]
        print(f"- {value:5.1f}% <- {galaxy_name}")


# def make_test():
#     dataset = MyDataset("test")
#     test_dl = DataLoader(dataset, batch_size=32)
#     predictions = trainer.test_epoch(test_dl)
#     predictions_new = []
#     for pred, f_name in predictions:
#         # From 5 to Barred Spiral
#         pred = trainer.dataset.galaxy_names[pred]
#         predictions_new.append((f_name, pred))

#     # Sort by filename
#     predictions_new = sorted(predictions_new, key=lambda x: x[0])

#     with open("output.csv", "w", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerows(predictions_new)
#     print("\nDone!")


os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
trainer = Trainer()
trainer.train()
print_accuracy()
