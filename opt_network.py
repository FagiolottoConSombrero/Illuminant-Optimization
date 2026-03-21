import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from utils import *
from torchmetrics.image import PeakSignalNoiseRatio, SpectralAngleMapper
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out


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
        elif model_type == 3:
            self.net = MST_Plus_Plus(in_channels=self.in_dim)

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

        w_illum = 3e-6
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


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


def shift_back(inputs,step=2):          # input [bs,28,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    down_sample = 256//row
    step = float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        inputs[:,i,:,:out_col] = \
            inputs[:,i,:,int(step*i):int(step*i)+out_col]
    return inputs[:, :, :, :out_col]


class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class MST(nn.Module):
    def __init__(self, in_dim=31, out_dim=31, dim=31, stage=2, num_blocks=[2,4,4]):
        super(MST, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSAB(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = MSAB(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        for (MSAB, FeaDownSample) in self.encoder_layers:
            fea = MSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            fea = LeWinBlcok(fea)

        # Mapping
        out = self.mapping(fea) + x

        return out


class MST_Plus_Plus(nn.Module):
    def __init__(self, in_channels=8, out_channels=121, n_feat=31, stage=3):
        super(MST_Plus_Plus, self).__init__()
        self.stage = stage
        self.n_feat = n_feat
        self.out_channels = out_channels

        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        modules_body = [
            MST(in_dim=n_feat, out_dim=n_feat, dim=n_feat, stage=2, num_blocks=[1, 1, 1])
            for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)

        # Conv di uscita ancora nello spazio feature (31 → 31)
        self.conv_out = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.to_spec = nn.Conv2d(n_feat, out_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        feat = self.conv_in(x)
        h = self.body(feat)
        h = self.conv_out(h)
        h = h + feat
        h = self.to_spec(h)
        return h[:, :, :h_inp, :w_inp]