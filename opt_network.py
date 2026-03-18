import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from utils import *
from torchmetrics.image import PeakSignalNoiseRatio, SpectralAngleMapper


class IlluminantOptimizer(nn.Module):
    """
    Ottimizza K illuminanti scegliendo, per ciascuno dei 15 LED,
    una tra 20 curve misurate.

    led_library: [15, 20, L]
    logits:      [K, 15, 20]
    output:      [K, L]
    """

    def __init__(self, num_illuminants=2, led_path=''):
        super().__init__()
        self.num_illuminants = num_illuminants
        self.register_buffer("led_library", load_led_library(mat_path=led_path))

        self.logits = nn.Parameter(
            torch.randn(num_illuminants, 15, 20) * 0.01
        )

    def forward(self, hard=False):
        """
        led_library : [15, 20, L]

        Returns:
        illuminants      : [K, L]
        """

        if hard:
            probs = F.gumbel_softmax(
                self.logits,
                tau=1,
                hard=True,
                dim=-1
            )   # [K,15,20]
        else:
            probs = F.softmax(self.logits / 1.0, dim=-1)

        # selezione soft delle curve
        # probs       [K,15,20]
        # led_library [15,20,L]
        # output      [K,15,L]
        selected_curves = torch.einsum("kic,icl->kil", probs, self.led_library)

        # somma dei 15 LED -> illuminante finale [K,L]
        illuminants = selected_curves.sum(dim=1)

        return illuminants


class IlluminantOptimizerL(nn.Module):
    """
    Ottimizza K illuminanti scegliendo, per ciascuno dei 15 LED,
    una tra 20 curve misurate.

    led_library: [15, 20, L]
    output:      [K, L]
    """

    def __init__(
        self,
        num_illuminants=2,
        led_path="",
        latent_dim=64,
        hidden_dim=128,
        temperature=1.0
    ):
        super().__init__()

        self.num_illuminants = num_illuminants
        self.temperature = temperature

        led_library = load_led_library(mat_path=led_path)   # [15,20,L]
        self.register_buffer("led_library", led_library)

        # un embedding learnable per ciascun illuminante
        self.illum_embedding = nn.Parameter(
            torch.randn(num_illuminants, latent_dim) * 0.02
        )

        # piccolo MLP che produce i logits [K,15,20]
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 15 * 20)
        )

    def forward(self, hard=False, return_probs=False, return_logits=False):
        """
        Returns
        -------
        illuminants : [K, L]
        """
        K = self.num_illuminants

        logits = self.mlp(self.illum_embedding).view(K, 15, 20)   # [K,15,20]

        if hard:
            probs = F.gumbel_softmax(
                logits,
                tau=self.temperature,
                hard=True,
                dim=-1
            )
        else:
            probs = F.softmax(logits / self.temperature, dim=-1)

        # [K,15,L]
        selected_curves = torch.einsum("kic,icl->kil", probs, self.led_library)

        # [K,L]
        illuminants = selected_curves.sum(dim=1)

        out = [illuminants]

        if return_probs:
            out.append(probs)
        if return_logits:
            out.append(logits)

        if len(out) == 1:
            return out[0]
        return tuple(out)

# =========================================================
# MDPM
# =========================================================

class MDPM(nn.Module):
    """
    Multi-path Deep Residual Module

    - preprocessing conv
    - 3 rami paralleli: 1x1, 3x3, 3x3 dilated(rate=4)
    - concat + fusion conv
    - residual add
    """
    def __init__(self, channels: int):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.path1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.path2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.path3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.pre(x)
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3 = self.path3(x)

        x = torch.cat([p1, p2, p3], dim=1)
        x = self.fuse(x)

        return x + residual


# =========================================================
# CRM
# =========================================================

class CRM(nn.Module):
    """
    Context Channel-wise Recalibration Module

    - concat feature corrente con input multi-image iniziale
    - channel attention stile squeeze-excitation
    - conv di fusione finale
    """
    def __init__(self, feat_channels: int, input_channels: int, reduction: int = 8):
        super().__init__()
        self.total_channels = feat_channels + input_channels

        hidden = max(self.total_channels // reduction, 8)

        self.fuse = nn.Sequential(
            nn.Conv2d(self.total_channels, feat_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Linear(self.total_channels, hidden)
        self.fc2 = nn.Linear(hidden, self.total_channels)

    def forward(self, feat: torch.Tensor, init_input: torch.Tensor) -> torch.Tensor:
        """
        feat:       [B, C_feat, H, W]
        init_input: [B, C_in,   H, W]
        """
        x = torch.cat([feat, init_input], dim=1)   # [B, C_feat + C_in, H, W]

        # channel attention
        w = F.adaptive_avg_pool2d(x, 1).flatten(1)      # [B, C]
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w)).view(x.shape[0], x.shape[1], 1, 1)

        x = x * w
        x = self.fuse(x)

        return x


# =========================================================
# MECM
# =========================================================

class MECM(nn.Module):
    """
    Multi-path Enhanced Calibration Module
    = MDPM + CRM
    """
    def __init__(self, feat_channels: int, input_channels: int):
        super().__init__()
        self.mdpm = MDPM(feat_channels)
        self.crm = CRM(feat_channels, input_channels)

    def forward(self, feat: torch.Tensor, init_input: torch.Tensor) -> torch.Tensor:
        feat = self.mdpm(feat)
        feat = self.crm(feat, init_input)
        return feat


# =========================================================
# SRNet
# =========================================================

class SRNet(nn.Module):
    """
    Deep spectral reconstruction backbone
    composto da 5 MECM
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 31,
        feat_channels: int = 64,
        num_mecm: int = 5
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.ModuleList([
            MECM(feat_channels=feat_channels, input_channels=in_channels)
            for _ in range(num_mecm)
        ])

        self.tail = nn.Conv2d(feat_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        init_input = x
        feat = self.head(x)

        for blk in self.blocks:
            feat = blk(feat, init_input)

        out = self.tail(feat)
        return out


class SpectralMLP(nn.Module):
    """
    Input:  (B, K, H, W)
    Output: (B, 31, H, W)
    MLP per-pixel: R^K -> R^31 applicato su ogni (h,w).
    """
    def __init__(self, hidden_dim=256, num_layers=3, out_activation=None, in_dim=6):
        super().__init__()
        act = nn.ReLU()

        layers = []
        for _ in range(max(0, num_layers - 1)):
            layers += [nn.Linear(in_dim, hidden_dim), act]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 31))
        if out_activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif out_activation == "softplus":
            layers.append(nn.Softplus(beta=1.0, threshold=20.0))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()   # (B,H,W,C)
        x = x.view(B * H * W, C)                 # (BHW, C)
        y = self.mlp(x)                          # (BHW, 31)
        y = y.view(B, H, W, 31).permute(0, 3, 1, 2).contiguous()  # (B,31,H,W)
        return y


class JointNetwork(pl.LightningModule):
    def __init__(self, lr=1e-3, patience=50, model_type=1, n_ill=2, in_dim=6, lambda_ang=0.2, led_path='', camera_spd_path=''):
        super().__init__()
        self.model_type = model_type
        self.save_hyperparameters()
        self.lr = lr
        self.n_ill = n_ill
        self.patience = patience
        self.in_dim = in_dim
        self.lambda_ang = lambda_ang
        self.led_path = led_path
        self.camera_spd_path = camera_spd_path
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
        self.sam_metric = SpectralAngleMapper()

        self.ill_optimizer = IlluminantOptimizerL(num_illuminants=self.n_ill, led_path=self.led_path)
        if model_type == 1:
            self.net = SRNet(in_channels=self.in_dim)  # poi clamp nella loss
        elif model_type == 2:
            self.net = SpectralMLP(in_dim=self.in_dim)

    def forward(self, x):
        # 1. Obtain Illuminants SPD
        ills = self.ill_optimizer()
        # 2.. Generate RGB Images
        rgb1, rgb2 = render_rgb(x, ills, self.camera_spd_path)
        in_rgb = torch.cat((rgb1, rgb2), dim=1)
        return self.net(in_rgb), ills, rgb1, rgb2

    def step(self, batch, stage):
        ref = batch  # [B,31,H,W]

        recon, ills, rgb1, rgb2 = self(ref)

        # reconstruction loss
        loss_rec = reconstruction_loss(recon, ref)

        # illuminant regularization
        loss_illum, _ = illumination_spec_regularization(ills)

        # img regularization
        loss_img, _ = illumination_img_regularization(rgb1, rgb2)

        w_illum = 3e-7
        w_img = 1e-5

        loss = loss_rec + w_illum * loss_illum + w_img * loss_img

        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, batch_size=ref.size(0))

        return loss, recon, ref

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon, ref = self.step(batch, "val")

        # ogni 10 epoch calcola PSNR e SSIM
        if self.current_epoch % 10 == 0:
            recon_eval = recon.clamp(0, 1)
            ref_eval = ref.clamp(0, 1)

            sam_val = self.sam_metric(recon_eval, ref_eval)
            psnr_val = self.psnr_metric(recon_eval, ref_eval)
            ssim_val = spectral_ssim(recon_eval, ref_eval)

            self.log("val_sam", sam_val, on_epoch=True, prog_bar=True, batch_size=ref.size(0))
            self.log("val_psnr", psnr_val, on_epoch=True, prog_bar=True, batch_size=ref.size(0))
            self.log("val_ssim", ssim_val, on_epoch=True, prog_bar=True, batch_size=ref.size(0))

            # stampa solo una volta per epoch
            if batch_idx == 0:
                print(
                    f"[Epoch {self.current_epoch}] "
                    f"val_sam={sam_val.item():.4f}, "
                    f"val_psnr={psnr_val.item():.4f}, "
                    f"val_ssim={ssim_val.item():.4f}"
                )

        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=self.patience
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss"
            }}


def reconstruction_loss(pred, target):
    return F.l1_loss(pred, target)


class IllNetwork(pl.LightningModule):
    def __init__(self, lr=1e-3, patience=50, n_ill=2, led_path='', camera_spd_path=''):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.n_ill = n_ill
        self.patience = patience
        self.led_path = led_path
        self.camera_spd_path = camera_spd_path

        self.ill_optimizer = IlluminantOptimizerL(num_illuminants=self.n_ill, led_path=self.led_path)

    def forward(self, x):
        # 1. Obtain Illuminants SPD
        ills = self.ill_optimizer()
        # 2.. Generate RGB Images
        rgb1, rgb2 = render_rgb(x, ills, self.camera_spd_path)
        return ills, rgb1, rgb2

    def step(self, batch, stage):
        ref = batch  # [B,31,H,W]

        ills, rgb1, rgb2 = self(ref)

        # illuminant regularization
        loss_illum, _ = illumination_spec_regularization(ills)

        # img regularization
        loss_img, _ = illumination_img_regularization(rgb1, rgb2)

        loss = loss_illum + loss_img

        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, batch_size=ref.size(0))

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=self.patience
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss"
            }}
