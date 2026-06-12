import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Patch64TinyClassifier(nn.Module):
    """
    Classificatore CNN minimale per patch HSI 64x64.

    Input:
        x: [B, C, 64, 64]

    Output:
        logits: [B, num_classes]
    """

    def __init__(
        self,
        in_channels=31,
        num_classes=15,
        spectral_width=4,
        spatial_width=8,
        dropout=0.6
    ):
        super().__init__()

        self.model = nn.Sequential(
            # Mixing spettrale molto compresso: 31 -> 4 feature
            nn.Conv2d(in_channels, spectral_width, kernel_size=1, bias=False),
            nn.GroupNorm(1, spectral_width),
            nn.GELU(),

            # Riduzione spaziale forte: 64x64 -> 16x16
            nn.AvgPool2d(kernel_size=4),

            # Convoluzione depthwise: poco costo, poco rischio overfitting
            nn.Conv2d(
                spectral_width,
                spectral_width,
                kernel_size=3,
                padding=1,
                groups=spectral_width,
                bias=False
            ),
            nn.GroupNorm(1, spectral_width),
            nn.GELU(),

            # Piccolo mixing tra feature
            nn.Conv2d(spectral_width, spatial_width, kernel_size=1, bias=False),
            nn.GroupNorm(1, spatial_width),
            nn.GELU(),

            nn.Dropout2d(dropout),

            # Descriptor globale della patch
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Dropout(dropout),
            nn.Linear(spatial_width, num_classes)
        )

    def forward(self, x):
        return self.model(x)

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

        self.net = Patch64TinyClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            spectral_width=4,
            spatial_width=8,
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