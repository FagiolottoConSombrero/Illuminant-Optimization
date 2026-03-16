import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from utils import *


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
        latent_dim=128,
        hidden_dim=256,
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

        self.ill_optimizer = IlluminantOptimizerL(num_illuminants=self.n_ill, led_path=self.led_path)
        if model_type == 1:
            self.net = SpectralMLP(in_dim=self.in_dim)  # poi clamp nella loss

    def forward(self, x):
        # 1. Obtain Illuminants SPD
        ills = self.ill_optimizer()
        # 2.. Generate RGB Images
        rgb1, rgb2 = render_rgb(x, ills, self.camera_spd_path)
        in_rgb = torch.cat((rgb1, rgb2), dim=1)
        return self.net(in_rgb), ills

    def step(self, batch, stage):
        ref = batch  # [B,31,H,W]

        recon, ills = self(ref)

        # reconstruction loss
        loss_rec = reconstruction_loss(recon, ref)

        # illuminant regularization
        loss_illum, _ = illumination_spec_regularization(ills)

        loss = loss_rec + loss_illum

        # metric
        mrae_val = mrae(recon, ref)

        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, batch_size=ref.size(0))
        self.log(f"{stage}_mrae", mrae_val, on_epoch=True, prog_bar=True, batch_size=ref.size(0))

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


def reconstruction_loss(pred, target):
    return F.l1_loss(pred, target)



