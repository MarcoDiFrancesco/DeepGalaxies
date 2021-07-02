import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from Dataset import MyDataset
from Network import Net
from time import strftime, gmtime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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

    def __init__(self, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # New model checkpoint
        self.log_dir = Path("logs") / strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.log_dir.mkdir(parents=True)
        self.writer = SummaryWriter(self.log_dir)

        if model_path is not None:
            assert os.path.isfile(model_path)
            self.model = torch.load(model_path)
            print("Using pre-trained weights")
        else:
            self.model = Net().to(self.device)
            self.model = nn.DataParallel(self.model)
            print("Training from scratch")
        self.lr = 1e-3
        self.loss_fn = nn.CrossEntropyLoss()
        # TODO: TRY ADAM
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr
        )  # weight_decay=1e-5
        self.best_accuracy = 0
        self.epochs = 200
        self.dataset = MyDataset("train")
        # Dataset 80 / 20 split
        self.train_ds, self.valid_ds = self.dataset.get_split(0.8)
        self.train_dl = DataLoader(
            self.train_ds, batch_size=32, shuffle=True, num_workers=4
        )
        self.valid_dl = DataLoader(self.valid_ds, batch_size=32, num_workers=4)

    def train(self):
        for epoch in range(self.epochs):
            accuracy_t, loss_t = self._train_epoch(self.train_dl)
            accuracy_v, loss_v, _ = self._valid_epoch(self.valid_dl)
            if self.best_accuracy < accuracy_v:
                torch.save(self.model, self.log_dir / "best.pth")
                self.best_accuracy = accuracy_v
            self.writer.add_scalars(
                "Loss",
                {"test": loss_v, "train": loss_t},
                epoch,
            )
            self.writer.add_scalars(
                "Accuracy",
                {"test": accuracy_v, "train": accuracy_t},
                epoch,
            )
            print(
                f"Epoch {epoch+1:>3}/{self.epochs} - Train accuracy {accuracy_t:5.2f}% loss {loss_t:8f} - Valid accuracy {accuracy_v:5.2f}% loss {loss_v:8f}"
            )

    def _train_epoch(self, dl):
        size = len(dl.dataset)
        loss_value, correct = 0, 0
        for images, labels in tqdm(dl):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(images)
            # Calculate loss
            loss = self.loss_fn(pred, labels)
            loss_value += loss.item()
            # Compare predictions with real values
            # argmax: [32, 10] -> [32] taking the most likely value
            values = pred.argmax(1) == labels
            # Get how many are true
            correct += values.sum().item()
            # Backpropagation
            loss.backward()
            self.optimizer.step()
        accuracy = (correct / size) * 100
        loss_value /= size
        return accuracy, loss_value

    def _valid_epoch(self, dl):
        # Switch layers
        self.model.eval()
        loss, correct = 0, 0
        size_by_label = {i: 0.0 for i in range(10)}
        correct_by_label = {i: 0.0 for i in range(10)}
        with torch.no_grad():
            for images, labels in tqdm(dl):
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.model(images)
                loss += self.loss_fn(preds, labels).item()
                # Compare predictions with real values
                for pred, label in zip(preds.argmax(1), labels):
                    # Get tensor value
                    label = label.item()
                    size_by_label[label] += 1
                    if pred == label:
                        # Compute per label accuracy
                        correct_by_label[label] += 1
                        correct += 1
        size = len(dl.dataset)
        accuracy = (correct / size) * 100
        loss /= size
        for key in correct_by_label.keys():
            correct_by_label[key] /= size_by_label[key] if size_by_label[key] else 1
            # To percentage
            correct_by_label[key] *= 100
        return accuracy, loss, correct_by_label

    def test_epoch(self, dl):
        """Testing without labels from pretrained model"""
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(dl):
                images = images.to(self.device)
                prediction = self.model(images)
                for pred, idx in zip(prediction.argmax(1), labels):
                    pred = pred.item()
                    # Get name from int
                    pred = self.dataset.galaxy_names[pred]
                    # String to int to sort correctly
                    idx = int(idx)
                    predictions.append((idx, pred))
        # Sort by filename
        predictions = sorted(predictions, key=lambda x: x[0])
        return predictions

    def extract_features(self, dl):
        features = []
        self.model.eval()
        with torch.no_grad():
            for images, _ in tqdm(dl):
                images = images.to(self.device)
                res = self.model.feature_extraction(images)
                features.append(res)
        return features
