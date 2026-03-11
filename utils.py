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
    illuminants = illuminants.to(device='gpu', dtype=reflectance.dtype)
    camera_spd = camera_spd.to(device='gpu', dtype=reflectance.dtype)
    response = illuminants[:, None, :] * camera_spd[None, :, :]

    # somma spettrale
    rgb_multi = torch.einsum("blhw,kcl->bkchw", reflectance, response)
    rgb1 = rgb_multi[:, 0]  # [B, 3, H, W]
    rgb2 = rgb_multi[:, 1]  # [B, 3, H, W]

    return rgb1, rgb2

