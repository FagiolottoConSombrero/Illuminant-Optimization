import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scipy.io
from torchmetrics.image import StructuralSimilarityIndexMeasure


def load_led_library(mat_path, device="cpu", dtype=torch.float32):
    data = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    light = data["light"]
    led_names = list(light._fieldnames)
    led_curves = []

    for name in led_names:
        M = getattr(light, name)   # [401,20]
        M = np.asarray(M, dtype=np.float32)
        led_curves.append(M.T)   # [20,401]

    led_library = np.stack(led_curves, axis=0)  # [15,20,401]
    wavelengths = np.linspace(380, 780, 401, dtype=np.float32)

    # CROP 400–700
    mask = (wavelengths >= 400) & (wavelengths <= 700)
    led_library = led_library[:, :, mask]

    led_library = torch.tensor(led_library, dtype=dtype, device=device)

    return led_library


def load_camera_SPD(csv_path):
    """
    Legge un CSV con sensitività spettrali camera campionate tra
    400–700 nm a passi di 10 nm

    CSV formato atteso:
        wavelength,R,G,B
        400,...
        410,...
        ...
        700,...

    Parameters
    ----------
    csv_path : str
        percorso del file CSV

    Returns
    -------
    wavelengths : torch.Tensor [28]
        lunghezze d'onda 400–670 nm

    camera_sens : torch.Tensor [3,28]
        sensitività R,G,B
    """

    df = pd.read_csv(csv_path)

    R = df.iloc[:, 1].values.astype(np.float32)
    G = df.iloc[:, 2].values.astype(np.float32)
    B = df.iloc[:, 3].values.astype(np.float32)

    # stack efficiente
    camera_np = np.stack([R, G, B], axis=0)
    camera_sens = torch.tensor(camera_np)

    return camera_sens


def render_rgb(reflectance, illuminants, camera_sens="/Users/kolyszko/Documents/NIKON-D810.csv"):
    """
    reflectance : [B, 31, H, W]
    illuminants : [K, 301]
    camera_sens : [3, 31]

    return:
        rgb_multi   : [B, K, 3, H, W]
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    camera_spd = load_camera_SPD(camera_sens)
    illuminants = illuminants[:, ::10]  # ill[K, 301] ---> ill[K, 31]
    # risposta spettrale combinata: [K, 3, 31]
    illuminants = illuminants.to(device=device, dtype=reflectance.dtype)
    camera_spd = camera_spd.to(device=device, dtype=reflectance.dtype)
    response = illuminants[:, None, :] * camera_spd[None, :, :]

    # somma spettrale
    rgb_multi = torch.einsum("blhw,kcl->bkchw", reflectance, response)
    rgb1 = rgb_multi[:, 0]  # [B, 3, H, W]
    rgb2 = rgb_multi[:, 1]  # [B, 3, H, W]

    return rgb1, rgb2


def normalize_illuminants(illuminants, eps=1e-8):
    """
    illuminants: [K, L]
    Normalizzazione per energia totale, come nel paper che parla di
    normalized SPD per E^(1), E^(2).
    """
    return illuminants / (illuminants.sum(dim=-1, keepdim=True) + eps)


def illumination_diversity_loss(illuminants, eps=1e-6):
    """
    Ldiv del paper, adattata al caso K=2.

    illuminants: [2, L]

    Ritorna una loss da minimizzare:
        mean( 1 / (|E1 - E2| + eps) )
    """
    if illuminants.shape[0] != 2:
        raise ValueError(f"Mi aspetto 2 illuminanti, ma ho shape {tuple(illuminants.shape)}")

    E = normalize_illuminants(illuminants, eps=eps)
    diff = torch.abs(E[0] - E[1])  # [L]
    loss = torch.mean(1.0 / (diff + eps))
    return loss


def spectral_ssim(recon, ref, data_range=1.0):
    """
    SSIM medio banda per banda.
    recon, ref: [B, C, H, W]
    """
    metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(recon.device)

    vals = []
    for c in range(recon.shape[1]):
        ssim_c = metric(recon[:, c:c+1], ref[:, c:c+1])
        vals.append(ssim_c)

    return torch.stack(vals).mean()

def illumination_diversity_loss(combined_illum_list):
    loss = 0.0
    num_pairs = 0
    for i in range(len(combined_illum_list)):
        for j in range(i + 1, len(combined_illum_list)):
            diff = torch.abs(combined_illum_list[i] - combined_illum_list[j])
            loss += 1.0 / (diff.mean() + 1e-6)
            num_pairs += 1
    return loss / (num_pairs + 1e-6)

def illumination_trend_dissimilarity_loss(combined_illum_list):
    """
    Encourage different trend (derivative) across combined illuminations.
    """
    loss = 0.0
    num_pairs = 0
    for i in range(len(combined_illum_list)):
        for j in range(i + 1, len(combined_illum_list)):
            d1 = torch.diff(combined_illum_list[i], dim=0)
            d2 = torch.diff(combined_illum_list[j], dim=0)
            diff = torch.abs(d1 - d2)
            loss += 1.0 / (diff.mean() + 1e-6)
            num_pairs += 1
    return loss / (num_pairs + 1e-6)

def illumination_nonzero_response_loss(combined_illum_list, min_thresh=0.15, max_thresh=1.8):
    """
    Penalize too-small responses and (lightly) very large responses.
    """
    loss = 0.0
    for illum in combined_illum_list:
        low_mask = illum < min_thresh
        low_loss = (1.0 / (illum + 1e-6)) * low_mask.float()
        # high_loss = F.relu(illum - max_thresh)  # optional soft cap
        loss += low_loss.mean()
    return loss / len(combined_illum_list)


def rgb_to_lab(rgb):
    """
    rgb: (B, 3, H, W) in [0,1] -> Lab
    """
    rgb = rgb.clamp(0, 1)
    return kcolor.rgb_to_lab(rgb)

def rgb_to_chromaticity(rgb):
    intensity = rgb.sum(dim=1, keepdim=True) + 1e-6
    return rgb / intensity

def illumination_render_difference_pixelwise_loss(rgb_out_list):
    loss = 0.0
    num_pairs = 0
    for i in range(len(rgb_out_list)):
        for j in range(i + 1, len(rgb_out_list)):
            diff = torch.abs(rgb_out_list[i] - rgb_out_list[j])
            loss += 1.0 / (diff.mean() + 1e-6)
            num_pairs += 1
    return loss / (num_pairs + 1e-6)

def illumination_render_difference_color_loss(rgb_out_list):
    loss = 0.0
    num_pairs = 0
    for i in range(len(rgb_out_list)):
        for j in range(i + 1, len(rgb_out_list)):
            rgb_i_lab = rgb_to_lab(rgb_out_list[i])
            rgb_j_lab = rgb_to_lab(rgb_out_list[j])
            diff = torch.abs(rgb_i_lab - rgb_j_lab)
            loss += 1.0 / (diff.mean() + 1e-6)
            num_pairs += 1
    return loss / (num_pairs + 1e-6)

def illumination_render_difference_chromaticity_loss(rgb_out_list):
    loss = 0.0
    num_pairs = 0
    for i in range(len(rgb_out_list)):
        for j in range(i + 1, len(rgb_out_list)):
            chroma_i = rgb_to_chromaticity(rgb_out_list[i])
            chroma_j = rgb_to_chromaticity(rgb_out_list[j])
            diff = torch.abs(chroma_i - chroma_j)
            loss += 1.0 / (diff.mean() + 1e-6)
            num_pairs += 1
    return loss / (num_pairs + 1e-6)


class IlluminationRegularizationLoss(nn.Module):
    """
    Composite regularizer over combined illumination spectra and rendered RGBs.
    """

    def __init__(self, alpha=1.0, gamma=2.0, delta=1.0, eta=1.0, zeta=1.0, theta=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.eta = eta
        self.zeta = zeta
        self.theta = theta

    def forward(self, combined_illum_list, rgb_out_list):
        loss = 0.0
        #Illuminants losses
        loss += self.alpha * illumination_diversity_loss(combined_illum_list)
        loss += self.theta * illumination_trend_dissimilarity_loss(combined_illum_list)
        loss += self.gamma * illumination_nonzero_response_loss(combined_illum_list)
        #RGBs losses
        loss += self.delta * illumination_render_difference_pixelwise_loss(rgb_out_list)
        loss += self.eta * illumination_render_difference_color_loss(rgb_out_list)
        loss += self.zeta * illumination_render_difference_chromaticity_loss(rgb_out_list)
        return loss
    

