from Trainer import Trainer
import os
from pathlib import Path


def print_accuracy(trainer: Trainer):
    """Print 3 metrics given a Trainer: accuracy, loss, label-wise accuracy

    Args:
        trainer (Trainer): [description]
    """
    print("--- Train dataset ---")
    acc, loss, accuracy_by_label = trainer.valid_epoch(trainer.train_dl)
    _print_stats(acc, loss, accuracy_by_label)
    print("--- Valid dataset ---")
    acc, loss, accuracy_by_label = trainer.valid_epoch(trainer.valid_dl)
    _print_stats(acc, loss, accuracy_by_label)


def _print_stats(acc: float, loss: float, accuracy_by_label: dict):
    """Print train and validation performances by galaxy

    Args:
        acc (float): accuracy
        loss (float): loss
        accuracy_by_label (dict): accuracy by label
    """
    print(f"Accuracy {acc:.2f}%    ")
    print(f"Loss {loss:8f}")
    for key, value in accuracy_by_label.items():
        galaxy_name = trainer.train_ds.galaxy_names[key]
        print(f"- {value:5.1f}% <- {galaxy_name}")


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
model_path = (
    Path("/")
    / "thunderdisk"
    / "data_rene_policistico_log"
    / "2021-08-23 11:51:08"
    / "535.pth"
)
trainer = Trainer(model_path)
# trainer.train()
print_accuracy(trainer)
