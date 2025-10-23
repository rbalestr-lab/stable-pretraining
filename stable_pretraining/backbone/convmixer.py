from torch import nn


class ConvMixer(nn.Module):
    """ConvMixer model.

    A simple and efficient convolutional architecture that operates directly on patches.

    Args:
        in_channels (int, optional): Number of input channels. Defaults to 3.
        num_classes (int, optional): Number of output classes. Defaults to 10.
        dim (int, optional): Hidden dimension size. Defaults to 64.
        depth (int, optional): Number of ConvMixer blocks. Defaults to 6.
        kernel_size (int, optional): Kernel size for depthwise convolution. Defaults to 9.
        patch_size (int, optional): Patch embedding size. Defaults to 7.

    Note:
        Introduced in :cite:`trockman2022patches`.
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        dim=64,
        depth=6,
        kernel_size=9,
        patch_size=7,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )

        self.blocks_a = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(),
                )
                for _ in range(depth)
            ]
        )
        self.blocks_b = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1), nn.BatchNorm2d(dim), nn.ReLU()
                )
                for _ in range(depth)
            ]
        )

        self.pool = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten())
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, xb):
        """Forward pass through the ConvMixer model.

        Args:
            xb (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        out = self.conv1(xb)
        for a, b in zip(self.blocks_a, self.blocks_b):
            out = out + a(out)
            out = b(out)
        out = self.fc(self.pool(out))
        return out
