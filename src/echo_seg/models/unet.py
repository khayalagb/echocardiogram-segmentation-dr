"""U-Net segmentation model via segmentation-models-pytorch."""

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(
    encoder: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 1,
    num_classes: int = 4,
) -> nn.Module:
    """Return a U-Net.

    encoder_weights=None trains from scratch; 'imagenet' fine-tunes a pretrained
    encoder (adapts the first conv to in_channels=1 automatically).
    """
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,  # raw logits; loss functions handle softmax
    )
