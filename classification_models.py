import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        stride=1,
        padding=None,
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

        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualConvBlock(nn.Module):
    """
    Blocco residuale standard:
        Conv 3x3 -> BN -> GELU
        Conv 3x3 -> BN
        residual
    """

    def __init__(self, channels, dropout=0.1):
        super().__init__()

        self.conv1 = ConvBNAct(
            channels,
            channels,
            kernel_size=3,
            act=True
        )

        self.conv2 = ConvBNAct(
            channels,
            channels,
            kernel_size=3,
            act=False
        )

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        res = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)

        return F.gelu(out + res)


class SpectralChannelAttention(nn.Module):
    """
    SE-like attention sulle feature.
    La terrei opzionale: può aiutare perché le bande/feature non hanno
    tutte la stessa importanza.
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


class ResidualAttentionBlock(nn.Module):
    def __init__(self, channels, dropout=0.1, use_attention=True):
        super().__init__()

        self.res_block = ResidualConvBlock(
            channels=channels,
            dropout=dropout
        )

        self.attn = SpectralChannelAttention(channels) if use_attention else nn.Identity()

    def forward(self, x):
        x = self.res_block(x)
        x = self.attn(x)
        return x


class HSITextureCNN_BN(nn.Module):
    """
    CNN 2D con BatchNorm per classificazione di patch HSI 64x64.

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
        dropout=0.25,
        use_attention=True
    ):
        super().__init__()

        # --------------------------------------------------
        # 64 x 64
        # --------------------------------------------------
        self.stem = nn.Sequential(
            # mixing spettrale iniziale
            ConvBNAct(
                in_channels,
                width,
                kernel_size=1,
                padding=0
            ),

            # feature spaziali locali
            ConvBNAct(
                width,
                width,
                kernel_size=3
            )
        )

        self.stage1 = nn.Sequential(
            ResidualAttentionBlock(
                width,
                dropout=dropout,
                use_attention=use_attention
            )
        )

        # --------------------------------------------------
        # 64 -> 32
        # --------------------------------------------------
        self.down1 = ConvBNAct(
            width,
            width * 2,
            kernel_size=3,
            stride=2
        )

        self.stage2 = nn.Sequential(
            ResidualAttentionBlock(
                width * 2,
                dropout=dropout,
                use_attention=use_attention
            )
        )

        # --------------------------------------------------
        # 32 -> 16
        # --------------------------------------------------
        self.down2 = ConvBNAct(
            width * 2,
            width * 4,
            kernel_size=3,
            stride=2
        )

        self.stage3 = nn.Sequential(
            ResidualAttentionBlock(
                width * 4,
                dropout=dropout,
                use_attention=use_attention
            )
        )

        # --------------------------------------------------
        # 16 -> 8
        # --------------------------------------------------
        self.down3 = ConvBNAct(
            width * 4,
            width * 4,
            kernel_size=3,
            stride=2
        )

        self.stage4 = nn.Sequential(
            ResidualAttentionBlock(
                width * 4,
                dropout=dropout,
                use_attention=use_attention
            )
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)

        x = self.down1(x)
        x = self.stage2(x)

        x = self.down2(x)
        x = self.stage3(x)

        x = self.down3(x)
        x = self.stage4(x)

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

        self.net = HSITextureCNN_BN(
            in_channels=in_channels,
            num_classes=num_classes,
            width=width,
            dropout=dropout,
            use_attention=True
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