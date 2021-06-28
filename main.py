from Trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import os
import csv


def plot_images():
    # Plot images by type
    images_dir = trainer.dataset.dest_dir
    img_by_type = {i: [] for i in range(10)}
    for img_name in os.listdir(images_dir):
        # e.g. 3
        img_type = int(img_name[:1])
        img_by_type[img_type].append(img_name)

    rows = 2  # 1 removes the dimention
    fig, ax = plt.subplots(rows, 10, figsize=(40, 4 * rows))
    fig.suptitle("Image examples")
    for i in range(10):
        for j in range(rows):
            img = img_by_type[i][j]
            img = np.load(images_dir / img)
            img = np.transpose(img, (2, 1, 0))
            title = i
            ax[j][i].set_title(title)
            ax[j][i].imshow(img)


def plot_stats():
    # Plot loss and accuracy
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 11))
    # Get image sample
    img_size = trainer.train_ds[0][0].shape
    fig.suptitle(f"VGG16 - Image size: {img_size} - LR: {trainer.lr}")
    epochs = list(range(1, trainer.epochs + 1))
    # epochs = list(range(1, 12))
    train, test = ax1.plot(epochs, trainer.losses, label="test")
    ax1.set_title("Loss")
    ax1.legend((train, test), ("Train", "Test"))
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    train, test = ax2.plot(epochs, trainer.accuracies, label="test2")
    ax2.set_title("Accuracy")
    ax2.legend((train, test), ("Train", "Test"))
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.show()


# Print train and validation perfornces by galaxy
def print_stats(acc, loss, accuracy_by_label):
    print(f"Accuracy {acc:.2f}%    ")
    print(f"Loss {loss:8f}")
    for key, value in accuracy_by_label.items():
        galaxy_name = trainer.dataset.galaxy_names[key]
        print(f"- {value:5.1f}% <- {galaxy_name}")


def print_accuracy():
    print(f"--- Train dataset ---")
    acc, loss, accuracy_by_label = trainer.valid_epoch(trainer.train_dl)
    print_stats(acc, loss, accuracy_by_label)
    print(f"--- Valid dataset ---")
    acc, loss, accuracy_by_label = trainer.valid_epoch(trainer.valid_dl)
    print_stats(acc, loss, accuracy_by_label)


def make_test():
    dataset = MyDataset("test")
    test_dl = DataLoader(dataset, batch_size=32)
    predictions = trainer.test_epoch(test_dl)
    predictions_new = []
    for pred, f_name in predictions:
        # From 5 to Barred Spiral
        pred = trainer.dataset.galaxy_names[pred]
        predictions_new.append((f_name, pred))

    # Sort by filename
    predictions_new = sorted(predictions_new, key=lambda x: x[0])

    with open("output.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(predictions_new)
    print("\nDone!")


trainer = Trainer()
print(trainer.galaxy_names)
plot_images()
trainer.train()
