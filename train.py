from Trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import os
import csv


def print_accuracy(trainer: Trainer):
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


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
trainer = Trainer()
trainer.train()
print_accuracy(trainer)
