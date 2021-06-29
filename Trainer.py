import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from Dataset import MyDataset
from Network import Net
from time import strftime, gmtime


class Trainer:
    """
    galaxy_names = {
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
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # New model checkpoint
        self.dir_weights = Path("logs") / strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.dir_weights.mkdir(parents=True)
        # # Old model checkpoint
        # model_path = Path("logs") / "kaggle" / "input" / "galaxies" / "model.ckpt"
        # # Load model checkpoint
        # if os.path.isfile(model_path):
        #     self.model = torch.load(model_path)
        #     print("Using pre-trained weights")
        # else:
        print("Training from scratch")
        self.model = Net().to(self.device)
        # Multi gpu support
        self.model = nn.DataParallel(self.model)
        self.lr = 1e-5
        self.loss_fn = nn.CrossEntropyLoss()
        # TODO: TRY ADAM
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr
        )  # weight_decay=1e-5
        self.epochs = 20
        self.dataset = MyDataset("train")
        # Dataset 80 / 20 split
        self.train_ds, self.valid_ds = self.dataset.get_split(0.8)
        self.train_dl = DataLoader(self.train_ds, batch_size=64, shuffle=True)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=64)
        # Stats and Checkpoints directory
        self.losses = []
        self.accuracies = []

    def train(self):
        for epoch in range(self.epochs):
            correct_t, loss_t = self.train_epoch(self.train_dl)
            correct_v, loss_v, _ = self.valid_epoch(self.valid_dl)
            torch.save(self.model, self.dir_weights / f"{epoch}.ckpt")
            self.losses.append((loss_t, loss_v))
            self.accuracies.append((correct_t, correct_v))
            print(
                f"Epoch {epoch+1:>3}/{self.epochs} - Train accuracy {correct_t:5.2f}% loss {loss_t:8f} - Valid accuracy {correct_v:5.2f}% loss {loss_v:8f}"
            )

    def train_epoch(self, dl: DataLoader):
        size = len(dl.dataset)
        loss_value, correct = 0, 0
        for batch, (inputs, labels) in enumerate(dl):
            print(f"Train batch: {batch:>3}/{len(dl):<3}", end="\r")
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(inputs)
            # Calculate loss
            loss = self.loss_fn(pred, labels)
            loss_value += loss.item()
            # Compare predictions with real values
            values = pred.argmax(1) == labels
            # Get how many are true
            correct += values.sum().item()
            # Backpropagation
            loss.backward()
            self.optimizer.step()
        accuracy = (correct / size) * 100
        loss_value /= size
        return accuracy, loss_value

    def valid_epoch(self, dl):
        # Switch layers
        self.model.eval()
        loss, correct = 0, 0
        size_by_label = {i: 0.0 for i in range(10)}
        correct_by_label = {i: 0.0 for i in range(10)}
        with torch.no_grad():
            # For each batch
            for batch, (inputs, labels) in enumerate(dl):
                print(f"Valid batch: {batch:>3}/{len(dl):<3}", end="\r")
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                pred = self.model(inputs)
                loss += self.loss_fn(pred, labels).item()
                # Compare predictions with real values
                for pred, real in zip(pred.argmax(1), labels):
                    # TODO: find a nicer way to get prediction from tensor
                    galaxy_num = int(real)
                    size_by_label[galaxy_num] += 1
                    if pred == real:
                        # Compute per label accuracy
                        correct_by_label[galaxy_num] += 1
                        correct += 1
        size = len(dl.dataset)
        accuracy = (correct / size) * 100
        loss /= size
        for key in correct_by_label.keys():
            correct_by_label[key] /= size_by_label[key] if size_by_label[key] else 1.0
            # To percentage
            correct_by_label[key] *= 100
        return accuracy, loss, correct_by_label

    def test_epoch(self, dl):
        """
        Test function
        """
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for batch, (inputs, f_names) in enumerate(dl):
                print(f"Test batch: {batch:>3}/{len(dl):<3}", end="\r")
                inputs = inputs.to(self.device)
                pred = self.model(inputs)
                for pred, f_name in zip(pred.argmax(1), f_names):
                    # TODO: find a nicer way to get prediction from tensor
                    pred = int(pred)
                    f_name = int(f_name)
                    predictions.append((pred, f_name))
        return predictions
