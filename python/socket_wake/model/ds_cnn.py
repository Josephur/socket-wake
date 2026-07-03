# SPDX-License-Identifier: Apache-2.0
"""DS-CNN-L: Depthwise Separable CNN, ~24K params, Google KWS baseline.

Architecture (4 DS blocks, 64 hidden channels, GAP head):

    Input (B, 1, n_mels, n_frames)
        |
    Conv2d(1 -> 64, 3x3) + BN + ReLU     # stem
        |
    [DS block] x 4                      # depthwise 3x3 + pointwise 1x1
        |
    AdaptiveAvgPool2d(1) + Linear(64 -> n_classes)

For inference we collapse the (n_frames, n_mels) into a single 2D input
and let the model treat it as a (1, n_mels, n_frames=1) image -- this
matches how the runtime would feed a stacked-mel frame to the kernel.
"""

from torch import nn


def _ds_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False),
        nn.BatchNorm2d(in_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_c, out_c, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class DSCNN(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            _ds_block(64, 64),
            _ds_block(64, 64),
            _ds_block(64, 64),
            _ds_block(64, 64),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # Accept (B, n_mels) and treat as (B, 1, n_mels, 1) at the stem.
        # For real training we feed (B, 1, n_mels, n_frames=10) directly.
        if x.dim() == 2:
            x = x.unsqueeze(1).unsqueeze(-1)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)