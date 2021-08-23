from Trainer import Trainer
import os
from pathlib import Path


def print_accuracy(trainer: Trainer):
    print("--- Train dataset ---")
    acc, loss, accuracy_by_label = trainer.valid_epoch(trainer.train_dl)
    _print_stats(acc, loss, accuracy_by_label)
    print("--- Valid dataset ---")
    acc, loss, accuracy_by_label = trainer.valid_epoch(trainer.valid_dl)
    _print_stats(acc, loss, accuracy_by_label)


# Print train and validation perfornces by galaxy
def _print_stats(acc: float, loss: float, accuracy_by_label: dict):
    print(f"Accuracy {acc:.2f}%    ")
    print(f"Loss {loss:8f}")
    for key, value in accuracy_by_label.items():
        galaxy_name = trainer.dataset.galaxy_names[key]
        print(f"- {value:5.1f}% <- {galaxy_name}")


os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
model_path = (
    Path("/")
    / "thunderdisk"
    / "data_rene_policistico_log"
    / "2021-08-23 07:21:09"
    / "90.pth"
)
trainer = Trainer()
trainer.train()
print_accuracy(trainer)
