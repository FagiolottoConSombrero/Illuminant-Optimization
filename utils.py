import torch
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
    camera_spd = load_camera_SPD(camera_sens)
    illuminants = illuminants[:, ::10]  # ill[K, 301] ---> ill[K, 31]
    # risposta spettrale combinata: [K, 3, 31]
    illuminants = illuminants.to(device='cuda', dtype=reflectance.dtype)
    camera_spd = camera_spd.to(device='cuda', dtype=reflectance.dtype)
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


def illumination_trend_loss(illuminants, eps=1e-6):
    """
    Ltrend del paper, adattata al caso K=2.

    illuminants: [2, L]

    DeltaE = E[:,1:] - E[:,:-1]
    loss = mean( 1 / (|DeltaE1 - DeltaE2| + eps) )
    """
    if illuminants.shape[0] != 2:
        raise ValueError(f"Mi aspetto 2 illuminanti, ma ho shape {tuple(illuminants.shape)}")

    E = normalize_illuminants(illuminants, eps=eps)

    dE1 = E[0, 1:] - E[0, :-1]   # [L-1]
    dE2 = E[1, 1:] - E[1, :-1]   # [L-1]

    diff_trend = torch.abs(dE1 - dE2)
    loss = torch.mean(1.0 / (diff_trend + eps))
    return loss


def illumination_nonzero_loss(illuminants, tau=0.15, eps=1e-6):
    """
    Lnonzero del paper, adattata al caso generale K illuminanti.

    illuminants: [K, L]

    Penalizza bande con intensità troppo bassa.
    Se E(k, l) < tau, aggiunge:
        1 / (E(k, l) + eps)

    Nota:
    - applichiamo la loss sugli illuminanti normalizzati, per coerenza
      con le altre due loss.
    - tau va quindi interpretata nello spazio normalizzato.
    """
    E = normalize_illuminants(illuminants, eps=eps)   # [K, L]

    mask = (E < tau).float()                          # [K, L]
    penalty = mask / (E + eps)                        # [K, L]

    loss = torch.mean(penalty)
    return loss


def illumination_spec_regularization(illuminants, tau=0.15, eps=1e-6):
    """
    Peso uguale per:
        Ldiv + Ltrend + Lnonzero

    illuminants: [2, L]
    """
    l_div = illumination_diversity_loss(illuminants, eps=eps)
    l_trend = illumination_trend_loss(illuminants, eps=eps)
    l_nonzero = illumination_nonzero_loss(illuminants, tau=tau, eps=eps)

    loss = l_div + l_trend + l_nonzero

    return loss, {
        "illum_div": l_div.detach(),
        "illum_trend": l_trend.detach(),
        "illum_nonzero": l_nonzero.detach(),
    }


def rgb_to_xyz(rgb):
    """
    rgb: [B,3,H,W] in range arbitrario non negativo

    Conversione lineare RGB -> XYZ.
    Assumiamo RGB linearizzato.
    """
    if rgb.shape[1] != 3:
        raise ValueError(f"Mi aspetto rgb con 3 canali, ma ho shape {tuple(rgb.shape)}")

    # matrice standard sRGB linear -> XYZ (D65)
    M = rgb.new_tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])  # [3,3]

    B, C, H, W = rgb.shape
    x = rgb.permute(0, 2, 3, 1).contiguous()   # [B,H,W,3]
    xyz = torch.matmul(x, M.t())               # [B,H,W,3]
    xyz = xyz.permute(0, 3, 1, 2).contiguous() # [B,3,H,W]

    return xyz


def xyz_to_lab(xyz, eps=1e-6):
    """
    xyz: [B,3,H,W]
    output: lab [B,3,H,W]

    White point D65
    """
    if xyz.shape[1] != 3:
        raise ValueError(f"Mi aspetto xyz con 3 canali, ma ho shape {tuple(xyz.shape)}")

    # white point D65
    white = xyz.new_tensor([0.95047, 1.00000, 1.08883]).view(1, 3, 1, 1)

    xyz_n = xyz / (white + eps)

    delta = 6 / 29
    delta3 = delta ** 3
    factor = 1 / (3 * delta ** 2)
    offset = 4 / 29

    def f(t):
        return torch.where(t > delta3, torch.pow(t.clamp_min(eps), 1/3), factor * t + offset)

    fx = f(xyz_n[:, 0:1])
    fy = f(xyz_n[:, 1:2])
    fz = f(xyz_n[:, 2:3])

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    lab = torch.cat([L, a, b], dim=1)
    return lab


def rgb_to_lab(rgb, eps=1e-6):
    """
    rgb: [B,3,H,W]
    """
    xyz = rgb_to_xyz(rgb)
    lab = xyz_to_lab(xyz, eps=eps)
    return lab


def pixel_response_difference_loss(rgb1, rgb2, eps=1e-6):
    """
    Lpix del paper:
    incoraggia differenze pixel-wise tra le due immagini RGB.

    rgb1, rgb2: [B,3,H,W]

    Loss da minimizzare:
        mean( 1 / (|rgb1-rgb2| + eps) )
    """
    diff = torch.abs(rgb1 - rgb2)
    loss = torch.mean(1.0 / (diff + eps))
    return loss


def perceptual_color_difference_loss(rgb1, rgb2, eps=1e-6):
    """
    Lcolor del paper:
    differenza percettiva nello spazio Lab.

    rgb1, rgb2: [B,3,H,W]

    Loss da minimizzare:
        mean( 1 / (|Lab(rgb1)-Lab(rgb2)| + eps) )
    """
    lab1 = rgb_to_lab(rgb1, eps=eps)
    lab2 = rgb_to_lab(rgb2, eps=eps)

    diff = torch.abs(lab1 - lab2)
    loss = torch.mean(1.0 / (diff + eps))
    return loss


def chromaticity_difference_loss(rgb1, rgb2, eps=1e-6):
    """
    Lchrom del paper:
    differenza di cromaticità dopo rimozione della luminanza.

    rgb1, rgb2: [B,3,H,W]

    chrom(rgb) = rgb / sum_c rgb_c

    Loss da minimizzare:
        mean( 1 / (|chrom1-chrom2| + eps) )
    """
    sum1 = rgb1.sum(dim=1, keepdim=True)
    sum2 = rgb2.sum(dim=1, keepdim=True)

    chrom1 = rgb1 / (sum1 + eps)
    chrom2 = rgb2 / (sum2 + eps)

    diff = torch.abs(chrom1 - chrom2)
    loss = torch.mean(1.0 / (diff + eps))
    return loss


def illumination_img_regularization(rgb1, rgb2, eps=1e-6):
    """
    Lillum-img = Lpix + Lcolor + Lchrom

    rgb1, rgb2: [B,3,H,W]
    """
    l_pix = pixel_response_difference_loss(rgb1, rgb2, eps=eps)
    l_color = perceptual_color_difference_loss(rgb1, rgb2, eps=eps)
    l_chrom = chromaticity_difference_loss(rgb1, rgb2, eps=eps)

    loss = l_pix + l_color + l_chrom

    return loss, {
        "illum_pix": l_pix.detach(),
        "illum_color": l_color.detach(),
        "illum_chrom": l_chrom.detach(),
    }


def spectral_ssim(recon, ref, data_range=1.0):
    """
    SSIM medio banda per banda.
    recon, ref: [B, C, H, W]
    """
    vals = []
    for c in range(recon.shape[1]):
        vals.append(
            StructuralSimilarityIndexMeasure(
                recon[:, c:c + 1],
                ref[:, c:c + 1],
                data_range=data_range
            )
        )
    return torch.stack(vals).mean()

