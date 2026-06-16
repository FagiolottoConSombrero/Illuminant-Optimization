import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class CReLU(nn.Module):
    def forward(self, x):
        return torch.cat([F.relu(x), F.relu(-x)], dim=1)


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        stride=1,
        padding=None,
        relu=True
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class HSITextureAllConvGrayNet(nn.Module):
    """
    Variante più fedele al paper:
        HSI [B, C, H, W]
        -> proiezione intensity/grayscale [B, 1, H, W]
        -> filter bank 11x11
        -> CReLU
        -> all-conv classifier
    """

    def __init__(
        self,
        in_channels=31,
        num_classes=15,
        n_filters=32,
        dropout=0.3,
        freeze_gray=False
    ):
        super().__init__()

        # --------------------------------------------------
        # HSI -> grayscale / intensity image
        # --------------------------------------------------
        self.to_gray = nn.Conv2d(
            in_channels,
            1,
            kernel_size=1,
            bias=False
        )

        # inizializzazione: media delle bande
        with torch.no_grad():
            self.to_gray.weight.fill_(1.0 / in_channels)

        if freeze_gray:
            for p in self.to_gray.parameters():
                p.requires_grad = False

        # --------------------------------------------------
        # Filter-bank 11x11 come nel paper
        # --------------------------------------------------
        self.filter_bank = nn.Conv2d(
            1,
            n_filters,
            kernel_size=11,
            stride=2,
            padding=5,
            bias=False
        )

        self.crelu = CReLU()
        self.bn0 = nn.BatchNorm2d(n_filters * 2)

        ch = n_filters * 2

        self.features = nn.Sequential(
            # 32 x 32
            ConvBNReLU(ch, 64, kernel_size=3, stride=1),
            nn.Dropout2d(dropout),

            # 32 -> 16
            ConvBNReLU(64, 96, kernel_size=3, stride=2),

            # 16 x 16
            ConvBNReLU(96, 96, kernel_size=3, stride=1),
            nn.Dropout2d(dropout),

            # 16 -> 8
            ConvBNReLU(96, 128, kernel_size=3, stride=2),

            # 8 x 8
            ConvBNReLU(128, 128, kernel_size=3, stride=1),
            nn.Dropout2d(dropout),

            ConvBNReLU(128, 192, kernel_size=3, stride=1),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
        )

    def forward(self, x):
        # x: [B, C, 64, 64]

        x = self.to_gray(x)       # [B, 1, 64, 64]
        x = self.filter_bank(x)   # [B, n_filters, 32, 32]
        x = self.crelu(x)         # [B, 2*n_filters, 32, 32]
        x = self.bn0(x)

        x = self.features(x)      # [B, 192, 8, 8]
        x = self.classifier(x)    # [B, num_classes, 8, 8]

        logits = x.mean(dim=(2, 3))
        return logits

class ClassificationNetwork(pl.LightningModule):
    def __init__(
        self,
        in_channels=31,
        num_classes=5,
        lr=1e-3,
        weight_decay=1e-4,
        patience=20,
        dropout=0.3,
        label_smoothing=0.0,
        freeze_gray=True,
        n_filters=32
    ):
        super().__init__()

        self.save_hyperparameters()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.freeze_gray = freeze_gray
        self.n_filters = n_filters

        self.net = HSITextureAllConvGrayNet(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            n_filters=self.n_filters,
            dropout=dropout,
            freeze_gray=self.freeze_gray
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

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,
            T_mult=2,
            eta_min=self.lr * 0.01
            )


        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }