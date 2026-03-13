import torch
import numpy as np
import pandas as pd
import scipy.io



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


def illumination_spec_regularization(illuminants, eps=1e-6):
    """
    Peso uguale per Ldiv e Ltrend.

    illuminants: [2, L]
    """
    l_div = illumination_diversity_loss(illuminants, eps=eps)
    l_trend = illumination_trend_loss(illuminants, eps=eps)

    loss = l_div + l_trend

    return loss, {"illum_div": l_div.detach(), "illum_trend": l_trend.detach()}


def mrae(pred, target, eps=1e-8):
    """
    Mean Relative Absolute Error

    pred   : [B,C,H,W]
    target : [B,C,H,W]
    """
    return torch.mean(torch.abs(pred - target) / (target + eps))

