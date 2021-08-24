from pathlib import Path
from matplotlib.pyplot import axis

import torch
from torch import nn
from torch.utils.data import DataLoader

from Dataset import MyDataset
from Network import Net, NetFeatureExtractor
from time import strftime, gmtime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np


class Trainer:
    def __init__(self, model_path: Path = None, feature_extraction=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.log_dir = (
            Path("/")
            / "thunderdisk"
            / "data_rene_policistico_log"
            / strftime("%Y-%m-%d %H:%M:%S", gmtime())
        )
        self.log_dir.mkdir(parents=True)
        self.writer = SummaryWriter(self.log_dir)
        if feature_extraction:
            # self.model = NetFeatureExtractor()
            self.model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50")
        else:
            # self.model = Net()
            self.model = torch.hub.load(
                "pytorch/vision:v0.10.0", "resnet50", pretrained=True
            )
        self.model = self.model.to(self.device)
        self.model = nn.DataParallel(self.model)
        if model_path is not None:
            assert model_path.exists(), f"Checkpoint does not exist: {model_path}"
            self.model.load_state_dict(torch.load(model_path))
            print("Using pre-trained weights")
        if feature_extraction:
            print("Removing layer", list(self.model.module.children())[-1])
            self.model.module = nn.Sequential(*list(self.model.module.children())[:-1])

        self.lr = 1e-4
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        self.epochs = 1000
        self.train_ds, self.valid_ds = MyDataset("train"), MyDataset("validation")

        # DEBUG: take only the first images
        # self.train_ds.images = self.train_ds.images[:200]
        # self.valid_ds.images = self.valid_ds.images[:200]

        batch_size = 128
        num_workers = 2
        self.train_dl = DataLoader(
            self.train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True
        )
        self.valid_dl = DataLoader(
            self.valid_ds, batch_size=batch_size, num_workers=num_workers
        )

    def train(self):
        for epoch in range(self.epochs):
            accuracy_t, loss_t = self._train_epoch(self.train_dl)
            accuracy_v, loss_v, _ = self.valid_epoch(self.valid_dl)
            if epoch % 1 == 0:  # 5
                torch.save(self.model.state_dict(), self.log_dir / f"{epoch}.pth")
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
        size = len(dl.dataset)
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
                    pred = dl.dataset.galaxy_names[pred]
                    # String to int to sort correctly
                    idx = int(idx)
                    predictions.append((idx, pred))
        # Sort by filename
        predictions = sorted(predictions, key=lambda x: x[0])
        return predictions

    def extract_features(self, dl):
        features = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for images, l in tqdm(dl):
                images = images.to(self.device)
                res = self.model(images)
                # [batch, 32768]
                features.append(res.cpu())
                labels.append(l)
        # From [iters, batch, 32768]
        # To [iters*batch, 32768]
        features = np.vstack(features)
        # From [iters, batch] to [iters*batch]
        labels = np.concatenate(labels)
        return features, labels
