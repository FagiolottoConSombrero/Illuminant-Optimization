import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class SpectralDropout(nn.Module):
    """
    Dropout sulle feature/bande canale-wise.
    Utile per evitare che il modello si appoggi troppo a poche bande.
    """
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        # x: [B, C, H, W]
        mask = torch.rand(
            x.size(0), x.size(1), 1, 1,
            device=x.device,
            dtype=x.dtype
        ) > self.p

        return x * mask / (1.0 - self.p)


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
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
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        num_groups = min(groups, out_ch)
        while out_ch % num_groups != 0:
            num_groups -= 1

        self.norm = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DepthwiseSeparableBlock(nn.Module):
    """
    Residual block leggero:
        depthwise 3x3 + pointwise 1x1

    Mantiene capacità spaziale, ma riduce molto i parametri rispetto
    a conv 3x3 dense.
    """
    def __init__(self, channels, dropout=0.2):
        super().__init__()

        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False
        )

        self.norm1 = nn.GroupNorm(1, channels)

        self.pw = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False
        )

        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        res = x

        out = self.dw(x)
        out = F.gelu(self.norm1(out))

        out = self.pw(out)
        out = self.norm2(out)

        out = self.dropout(out)

        return F.gelu(out + res)


class HSICompactResNetClassifier(nn.Module):
    """
    CNN compatta ma realistica per patch HSI 64x64.

    Input:
        x: [B, C, 64, 64]

    Output:
        logits: [B, num_classes]
    """

    def __init__(
        self,
        in_channels=31,
        num_classes=15,
        width=24,
        dropout=0.35,
        spectral_dropout=0.10
    ):
        super().__init__()

        self.spectral_dropout = SpectralDropout(p=spectral_dropout)

        # 64 x 64
        self.stem = nn.Sequential(
            ConvGNAct(in_channels, width, kernel_size=1, padding=0),
            ConvGNAct(width, width, kernel_size=3),
        )

        self.stage1 = nn.Sequential(
            DepthwiseSeparableBlock(width, dropout=dropout),
            DepthwiseSeparableBlock(width, dropout=dropout),
        )

        # 64 -> 32
        self.down1 = ConvGNAct(
            width,
            width * 2,
            kernel_size=3,
            stride=2
        )

        self.stage2 = nn.Sequential(
            DepthwiseSeparableBlock(width * 2, dropout=dropout),
            DepthwiseSeparableBlock(width * 2, dropout=dropout),
        )

        # 32 -> 16
        self.down2 = ConvGNAct(
            width * 2,
            width * 4,
            kernel_size=3,
            stride=2
        )

        self.stage3 = nn.Sequential(
            DepthwiseSeparableBlock(width * 4, dropout=dropout),
            DepthwiseSeparableBlock(width * 4, dropout=dropout),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, num_classes)
        )

    def forward(self, x):
        x = self.spectral_dropout(x)

        x = self.stem(x)
        x = self.stage1(x)

        x = self.down1(x)
        x = self.stage2(x)

        x = self.down2(x)
        x = self.stage3(x)

        logits = self.classifier(x)
        return logits

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

        self.net = HSICompactResNetClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            width=width,
            dropout=dropout,
            spectral_dropout=0.10
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