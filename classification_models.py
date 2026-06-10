import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=8,
        act=True
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        num_groups = min(groups, out_channels)

        while out_channels % num_groups != 0:
            num_groups -= 1

        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SpectralChannelAttention(nn.Module):
    """
    Channel attention sulle feature.
    Utile perché le bande/feature spettrali non hanno tutte la stessa importanza.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden = max(channels // reduction, 4)

        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.attn(x)
        return x * w


class ResidualBlock(nn.Module):
    def __init__(self, channels, expansion=1):
        super().__init__()

        hidden = channels * expansion

        self.block = nn.Sequential(
            ConvGNAct(channels, hidden, kernel_size=3),
            ConvGNAct(hidden, channels, kernel_size=3, act=False),
            SpectralChannelAttention(channels)
        )

    def forward(self, x):
        return F.gelu(x + self.block(x))


class HSITextureClassifier(nn.Module):
    """
    CNN 2D per classificazione di patch HSI.

    Input:
        x -> [B, C, H, W]

    Output:
        logits -> [B, num_classes]
    """

    def __init__(
        self,
        in_channels=31,
        num_classes=5,
        width=64,
        dropout=0.3
    ):
        super().__init__()

        # Mixing spettrale iniziale: combina le bande HSI
        self.spectral_projection = nn.Sequential(
            ConvGNAct(
                in_channels,
                width,
                kernel_size=1,
                padding=0
            ),
            ConvGNAct(
                width,
                width,
                kernel_size=3
            )
        )

        self.stage1 = nn.Sequential(
            ResidualBlock(width),
            ResidualBlock(width)
        )

        self.down1 = ConvGNAct(
            width,
            width * 2,
            kernel_size=3,
            stride=2
        )

        self.stage2 = nn.Sequential(
            ResidualBlock(width * 2),
            ResidualBlock(width * 2)
        )

        self.down2 = ConvGNAct(
            width * 2,
            width * 4,
            kernel_size=3,
            stride=2
        )

        self.stage3 = nn.Sequential(
            ResidualBlock(width * 4),
            ResidualBlock(width * 4)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, num_classes)
        )

    def forward_features(self, x):
        x = self.spectral_projection(x)

        x = self.stage1(x)

        x = self.down1(x)
        x = self.stage2(x)

        x = self.down2(x)
        x = self.stage3(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        logits = self.classifier(x)
        return logits
    

import torch
import torch.nn as nn
import pytorch_lightning as pl


class ClassificationNetwork(pl.LightningModule):
    def __init__(
        self,
        in_channels=31,
        num_classes=5,
        lr=1e-3,
        weight_decay=1e-4,
        patience=20,
        width=64,
        dropout=0.3,
        label_smoothing=0.0,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience

        self.net = HSITextureClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            width=width,
            dropout=dropout,
        )

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing
        )

    def forward(self, x):
        """
        x: HSI patch [B, C, H, W]

        Returns:
            logits: [B, num_classes]
        """
        logits = self.net(x)
        return logits

    def step(self, batch, stage):
        """
        batch:
            x: [B, C, H, W]
            y: [B]
        """

        x, y = batch

        logits = self(x)

        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )

        self.log(
            f"{stage}_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )

        return loss, logits, preds, y

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self.step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, preds, y = self.step(batch, "val")

        if batch_idx == 0:
            acc = (preds == y).float().mean()

            print(
                f"[Epoch {self.current_epoch}] "
                f"val_loss={loss.item():.6f} "
                f"val_acc={acc.item():.4f}"
            )

        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, preds, y = self.step(batch, "test")
        return {
            "test_loss": loss,
            "preds": preds,
            "targets": y,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=self.patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }