from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from ..config import MODEL_CONFIG


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=use_bias,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TrackNet(nn.Module):
    def __init__(self, output_channels: int = MODEL_CONFIG.output_channels) -> None:
        super().__init__()
        self.output_channels = output_channels

        self.conv1 = ConvBlock(9, 64)
        self.conv2 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv14 = ConvBlock(256, 128)
        self.conv15 = ConvBlock(128, 128)

        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv16 = ConvBlock(128, 64)
        self.conv17 = ConvBlock(64, 64)
        self.conv18 = ConvBlock(64, self.output_channels)

        self.softmax = nn.Softmax(dim=1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(
                    module.weight,
                    MODEL_CONFIG.weight_init_min,
                    MODEL_CONFIG.weight_init_max,
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, inference: bool = False) -> torch.Tensor:
        batch_size = x.size(0)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)

        x = self.up1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)

        x = self.up2(x)
        x = self.conv14(x)
        x = self.conv15(x)

        x = self.up3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)

        output = x.reshape(batch_size, self.output_channels, -1)

        if inference:
            output = self.softmax(output)

        return output


def load_model(
    path: Union[str, Path],
    device: str,
    output_channels: int = MODEL_CONFIG.output_channels,
) -> TrackNet:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model = TrackNet(output_channels=output_channels)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    return model
