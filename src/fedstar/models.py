"""FedSTAR's four-block audio encoder, classifier and contrastive objective."""

import torch
from torch import nn
from torch.nn import functional as F


class AudioBlock(nn.Module):
    def __init__(self, in_channels, width, dropout):
        super().__init__()
        self.temporal = nn.Sequential(
            # TensorFlow SAME puts the extra element on the right/bottom.
            nn.ZeroPad2d((1, 2, 0, 0)),
            nn.Conv2d(in_channels, width, (1, 4)),
            nn.GroupNorm(4, width, eps=1e-3),
            nn.ReLU(),
        )
        self.frequency = nn.Sequential(
            nn.ZeroPad2d((0, 0, 1, 2)),
            nn.Conv2d(in_channels, width, (4, 1)),
            nn.GroupNorm(4, width, eps=1e-3),
            nn.ReLU(),
        )
        self.joint = nn.Sequential(
            nn.Conv2d(2 * width, width, 1),
            nn.GroupNorm(4, width, eps=1e-3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.joint(torch.cat((self.temporal(x), self.frequency(x)), dim=1))


class AudioEncoder(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        channels = (1, 24, 32, 64, 128)
        self.blocks = nn.Sequential(
            *(AudioBlock(a, b, dropout) for a, b in zip(channels[:-1], channels[1:]))
        )
        self.apply(_initialize)

    def forward(self, x):
        return self.blocks(x).amax(dim=(-2, -1))


def _initialize(module):
    # Keras Conv2D and Dense defaults use Glorot uniform and zero biases.
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class AudioClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super().__init__()
        if num_classes < 2:
            raise ValueError("At least two classes are required")
        self.encoder = build_encoder(dropout)
        self.classifier = nn.Linear(128, num_classes)
        _initialize(self.classifier)

    def forward(self, x):
        """Return logits; average segment logits to classify a whole clip."""
        return self.classifier(self.encoder(x))


def build_encoder(dropout=0.1):
    return AudioEncoder(dropout)


def build_model(num_classes, dropout=0.1):
    return AudioClassifier(num_classes, dropout)


def regularization_loss(model, coefficient=0.0001):
    """Keras L2 convention: coefficient * sum(conv weights ** 2)."""
    return coefficient * sum(
        layer.weight.square().sum()
        for layer in model.modules()
        if isinstance(layer, nn.Conv2d)
    )


def contrastive_loss(first, second, bilinear):
    """COLA's anchor-to-positive InfoNCE with learned bilinear similarity.

    Each row's positive is the second crop of the same clip; the remaining
    positive-view clips are its negatives (paper reference [35], Eq. 2).
    """
    if first.ndim != 2 or first.shape != second.shape or first.shape[0] < 2:
        raise ValueError("Contrastive views require matching batches of at least two clips")
    logits = first @ bilinear @ second.T
    labels = torch.arange(first.shape[0], device=first.device)
    return F.cross_entropy(logits, labels)
