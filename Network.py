from torch import nn


class Operation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 3
        out_channels = 10
        self.down = nn.Sequential(
            Operation(in_channels, 64),
            Operation(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Operation(64, 128),
            Operation(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Operation(128, 256),
            Operation(256, 256),
            Operation(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Operation(256, 512),
            Operation(512, 512),
            Operation(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Operation(512, 512),
            Operation(512, 512),
            Operation(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fully_connected = nn.Sequential(
            # 512*8*8: channels*image size maxpolled 5 times
            # 4096: is fixed by the authors
            nn.Linear(512 * 8 * 8, 4096),
            nn.ReLU(True),  # In place True
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, out_channels),
        )

    def forward(self, x):
        x = self.down(x)
        # From [batch, 512, 8, 8] to [batch, 32768]
        x = x.reshape(x.shape[0], -1)
        x = self.fully_connected(x)
        return x
