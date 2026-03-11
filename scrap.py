import torch
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import h5py


def parse_led_wavelengths(led_names):
    """
    Esempio:
        ['x370', 'x390', 'x405', ...] -> [370, 390, 405, ...]
    """
    wavelengths = []
    for name in led_names:
        wl = int(name.replace("x", ""))
        wavelengths.append(wl)
    return wavelengths


def build_manual_choice_idx_bandwise(
    led_names,
    led_library,
    lambda_led,
    mode="blue_max_green_min",
    default_idx=0,
    device="cpu"
):
    """
    Sceglie per ogni LED la colonna migliore in base all'energia
    nella banda desiderata.

    mode:
        - "blue_max_green_min"
        - "red_blue_max_green_min"
    """

    led_wavelengths = parse_led_wavelengths(led_names)
    choice_idx = torch.full((len(led_names),), default_idx, dtype=torch.long, device=device)

    mid_idx = led_library.shape[1] // 2   # ≈ metà delle 20 curve

    for i, wl in enumerate(led_wavelengths):

        curves = led_library[i]   # [20, N_lambda]

        # --------------------------------------------------
        # LED verdi → minimo nel verde
        # --------------------------------------------------
        if 500 <= wl <= 570:

            mask = (lambda_led >= 500) & (lambda_led <= 570)
            band_energy = curves[:, mask].sum(dim=1)
            choice_idx[i] = torch.argmin(band_energy)

        # --------------------------------------------------
        # MODALITÀ: BLUE MAX
        # --------------------------------------------------
        elif mode == "blue_max_green_min":

            # blu → massimo
            if wl <= 480:

                mask = (lambda_led >= 400) & (lambda_led <= 480)
                band_energy = curves[:, mask].sum(dim=1)
                choice_idx[i] = torch.argmax(band_energy)

            # rosso → valore medio
            elif wl >= 600:

                choice_idx[i] = mid_idx

        # --------------------------------------------------
        # MODALITÀ: RED + BLUE MAX
        # --------------------------------------------------
        elif mode == "red_blue_max_green_min":

            # blu → massimo
            if wl <= 450:

                mask = (lambda_led >= 400) & (lambda_led <= 450)
                band_energy = curves[:, mask].sum(dim=1)
                choice_idx[i] = torch.argmax(band_energy)

            # rosso → massimo
            elif wl >= 600:

                mask = (lambda_led >= 600) & (lambda_led <= 700)
                band_energy = curves[:, mask].sum(dim=1)
                choice_idx[i] = torch.argmax(band_energy)

        else:
            raise ValueError(f"Mode non supportato: {mode}")

    return choice_idx

def build_hard_illuminant(choice_idx, led_library):
    """
    choice_idx   : [15]   indice scelto per ogni LED, valori 0..19
    led_library  : [15, 20, 401]

    return:
        E_401     : [401]
        selected  : [15, 401]
    """
    selected = []
    for i in range(15):
        selected.append(led_library[i, choice_idx[i]])  # [401]

    selected = torch.stack(selected, dim=0)   # [15, 401]
    E_401 = selected.sum(dim=0)               # [401]

    return E_401, selected


def build_interp_matrix(lambda_src, lambda_tgt, device="cpu", dtype=torch.float32):
    A = np.zeros((len(lambda_tgt), len(lambda_src)), dtype=np.float32)

    for i, x in enumerate(lambda_tgt):
        if x <= lambda_src[0]:
            A[i, 0] = 1.0
        elif x >= lambda_src[-1]:
            A[i, -1] = 1.0
        else:
            j = np.searchsorted(lambda_src, x) - 1
            x0, x1 = lambda_src[j], lambda_src[j + 1]
            w1 = (x - x0) / (x1 - x0)
            w0 = 1.0 - w1
            A[i, j] = w0
            A[i, j + 1] = w1

    return torch.tensor(A, dtype=dtype, device=device)


def crop_reflectance_to_400_670(reflectance):
    """
    reflectance: [B, 31, H, W]
    """
    return reflectance[:, :31, :, :]   # [B, 28, H, W]



def load_camera_sensitivities_400_670(csv_path):
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

    wavelengths = df.iloc[:, 0].values
    R = df.iloc[:, 1].values
    G = df.iloc[:, 2].values
    B = df.iloc[:, 3].values

    wavelengths = torch.tensor(wavelengths)
    camera_sens = torch.tensor([R, G, B])

    return wavelengths, camera_sens


def render_rgb(reflectance, illuminants, camera_sens):
    """
    reflectance : [B, 31, H, W]
    illuminants : [K, 31]
    camera_sens : [3, 31]

    return:
        rgb_multi   : [B, K, 3, H, W]
    """
    # risposta spettrale combinata: [K, 3, 28]
    response = illuminants[:, None, :] * camera_sens[None, :, :]

    # somma spettrale
    rgb_multi = torch.einsum("blhw,kcl->bkchw", reflectance, response)

    return rgb_multi


def forward_physical_rendering(choice_idx, led_library, reflectance, camera_sens_28, A):
    """
    choice_idx      : [15]
    led_library     : [15, 20, 401]
    reflectance     : [B, 31, H, W]
    camera_sens_28  : [3, 28]
    A               : [28, 401]

    return:
        rgb          : [B, 1, 3, H, W]
        E_401        : [401]
        E_28         : [28]
    """
    # 1. illuminante su 401 campioni
    E_401, _ = build_hard_illuminant(choice_idx, led_library)

    # 2. illuminante su 28 bande
    E_28 = A @ E_401

    # 3. riflettanza tagliata a 400-670
    reflectance_28 = crop_reflectance_to_400_670(reflectance)

    # 4. render RGB
    rgb = render_rgb(
        reflectance_28=reflectance_28,
        illuminants_28=E_28.unsqueeze(0),   # [1,28]
        camera_sens_28=camera_sens_28,
        delta_lambda=10.0
    )

    return rgb, E_401, E_28



def load_reflectance_h5(h5_path, device="cpu", dtype=torch.float32, add_batch_dim=True):
    """
    Legge un file .h5 con dataset:
        /spec
        /wvs

    Replica il comportamento MATLAB:
        data = h5read(path,'/spec');
        data = rot90(data, -1);
        wave = h5read(path,'/wvs');

    Restituisce i dati nel formato PyTorch:
        reflectance : [B, L, H, W]   se add_batch_dim=True
                      [L, H, W]      altrimenti
        wavelengths : [L]

    Note
    ----
    MATLAB rot90(data,-1) ruota di 90° in senso orario sui primi due assi.
    In NumPy l'equivalente è:
        np.rot90(data, -1, axes=(0,1))
    """

    with h5py.File(h5_path, "r") as f:
        if "/spec" not in f or "/wvs" not in f:
            raise KeyError("Nel file .h5 devono esserci i dataset '/spec' e '/wvs'")

        data = np.array(f["/spec"], dtype=np.float32)
        wave = np.array(f["/wvs"], dtype=np.float32).squeeze()

    # equivalente di MATLAB: rot90(data, -1)
    data = np.ascontiguousarray(np.rot90(data, -1, axes=(0, 1)))

    # ---------------------------------------------------------
    # Gestione shape
    # Vogliamo arrivare a [L, H, W] prima di aggiungere batch
    # ---------------------------------------------------------
    if data.ndim != 3:
        raise ValueError(f"Mi aspettavo un cubo 3D per /spec, ma ho shape {data.shape}")

    # Caso 1: già [H, W, L]
    if data.shape[-1] == len(wave):
        data = np.transpose(data, (2, 0, 1))   # -> [L, H, W]

    # Caso 2: già [L, H, W]
    elif data.shape[0] == len(wave):
        pass

    # Caso 3: magari [H, L, W]
    elif data.shape[1] == len(wave):
        data = np.transpose(data, (1, 0, 2))   # -> [L, H, W]

    else:
        raise ValueError(
            f"Non riesco a capire quale asse sia quello spettrale. "
            f"shape data={data.shape}, len(wave)={len(wave)}"
        )

    reflectance = torch.tensor(data, dtype=dtype, device=device)
    wavelengths = torch.tensor(wave, dtype=dtype, device=device)

    if add_batch_dim:
        reflectance = reflectance.unsqueeze(0)   # [1, L, H, W]

    return reflectance, wavelengths


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


def plot_selected_led_curves(choice_idx, led_library, led_names, lambda_led):
    plt.figure(figsize=(10, 5))
    for i, name in enumerate(led_names):
        curve = led_library[i, choice_idx[i]].detach().cpu().numpy()
        plt.plot(lambda_led.cpu().numpy(), curve, label=f"{name} -> col {int(choice_idx[i])}", alpha=0.8)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Power")
    plt.title("Selected curve for each LED")
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()

def inspect_led_peaks(led_library, led_names, lambda_led):
    """
    Stampa per ogni LED il picco della curva 19
    """
    print("\n=== Peak inspection ===")
    for i, name in enumerate(led_names):
        curve = led_library[i, 19].detach().cpu().numpy()  # curva 'massima'
        peak_idx = np.argmax(curve)
        peak_wl = lambda_led[peak_idx].item()
        peak_val = curve[peak_idx]
        print(f"{name:>5s} -> peak at {peak_wl:7.2f} nm, peak value = {peak_val:.6f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # --------------------------------------------------
    # 1. file paths
    # --------------------------------------------------
    mat_path = "/Users/kolyszko/Documents/MATLAB/Thouslite5.mat"
    camera_csv_path = "/Users/kolyszko/Documents/NIKON-D810.csv"

    # --------------------------------------------------
    # 2. carica libreria LED
    # --------------------------------------------------
    led_library, lambda_led, led_names = load_led_library(
        mat_path=mat_path,
        device=device,
        dtype=dtype
    )

    print("led_library shape:", led_library.shape)   # [15,20,401]
    print("LED names:", led_names)

    # --------------------------------------------------
    # 3. carica sensitività camera già croppate 400-670
    # --------------------------------------------------
    lambda_cam, camera_sens_28 = load_camera_sensitivities_400_670(
        csv_path=camera_csv_path,
        device=device,
        dtype=dtype
    )

    print("camera_sens_28 shape:", camera_sens_28.shape)  # [3,28]
    print("lambda_cam shape:", lambda_cam.shape)          # [28]

    # --------------------------------------------------
    # 4. costruisci matrice di interpolazione
    #    da 370:670 (401 campioni) -> 400:10:670 (28 bande)
    # --------------------------------------------------
    lambda_src = lambda_led.cpu().numpy()  # ora è già 400–700
    lambda_tgt = np.arange(400, 701, 10, dtype=np.float32)

    A = build_interp_matrix(
        lambda_src=lambda_src,
        lambda_tgt=lambda_tgt,
        device=device,
        dtype=dtype
    )

    print("A shape:", A.shape)   # [28,401]

    # --------------------------------------------------
    # 5. lettura reflectance reale da file .h5
    # --------------------------------------------------
    reflectance_path = "/Volumes/Lexar/31bands_h5/gtRef_3014.h5"

    reflectance, wavelengths_ref = load_reflectance_h5(
        h5_path=reflectance_path,
        device=device,
        dtype=dtype,
        add_batch_dim=True
    )

    print("reflectance shape:", reflectance.shape)  # [1,31,H,W] atteso
    print("wavelengths_ref shape:", wavelengths_ref.shape)  # [31]
    print("wavelengths_ref:", wavelengths_ref)

    # --------------------------------------------------
    # 6. esempio di scelta hard:
    #    un indice tra 0 e 19 per ciascuno dei 15 LED
    # --------------------------------------------------
    choice_idx = torch.randint(
        low=0,
        high=20,
        size=(15,),
        device=device
    )

    print("choice_idx shape:", choice_idx.shape)  # [15]
    print("choice_idx:", choice_idx)

    choice_idx_blue = build_manual_choice_idx_bandwise(
        led_names=led_names,
        led_library=led_library,
        lambda_led=lambda_led,
        mode="blue_max_green_min",
        device=device
    )

    choice_idx_red = build_manual_choice_idx_bandwise(
        led_names=led_names,
        led_library=led_library,
        lambda_led=lambda_led,
        mode="red_blue_max_green_min",
        device=device
    )

    # --------------------------------------------------
    # 7. simulazione fisica
    # --------------------------------------------------
    rgb_blue, E401_blue, E28_blue = forward_physical_rendering(
        choice_idx=choice_idx_blue,
        led_library=led_library,
        reflectance=reflectance,
        camera_sens_28=camera_sens_28,
        A=A,
    )

    rgb_red, E401_red, E28_red = forward_physical_rendering(
        choice_idx=choice_idx_red,
        led_library=led_library,
        reflectance=reflectance,
        camera_sens_28=camera_sens_28,
        A=A,
    )

    # --------------------------------------------------
    # 8. visualizza una RGB simulata del primo sample
    # --------------------------------------------------
    def show_rgb(rgb, title):
        rgb0 = rgb[0, 0].permute(1, 2, 0).detach().cpu().numpy()

        rgb0 = rgb0 - rgb0.min()
        if rgb0.max() > 0:
            rgb0 = rgb0 / rgb0.max()

        rgb0 = np.clip(rgb0, 0, 1)


        plt.figure(figsize=(5, 5))
        plt.imshow(rgb0)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------
    # 8.5 plot illuminanti
    # --------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(lambda_led.cpu().numpy(), E401_blue.detach().cpu().numpy(), label="Blue max / Green min")
    plt.plot(lambda_led.cpu().numpy(), E401_red.detach().cpu().numpy(), label="Red+Blue max / Green min")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Power")
    plt.title("Final illuminants on 370-670 nm grid")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(lambda_cam.cpu().numpy(), E28_blue.detach().cpu().numpy(), "o-", label="Blue max / Green min")
    plt.plot(lambda_cam.cpu().numpy(), E28_red.detach().cpu().numpy(), "o-", label="Red+Blue max / Green min")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Power")
    plt.title("Final illuminants on 400-670 nm grid")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    show_rgb(rgb_blue, "Blue max / Green min")
    show_rgb(rgb_red, "Red max / Green min")
    plot_selected_led_curves(choice_idx_red, led_library, led_names, lambda_led)
    print("\n=== Blue config ===")
    for name, idx in zip(led_names, choice_idx_blue.cpu().numpy()):
        print(f"{name}: column {idx}")

    print("\n=== Red+Blue config ===")
    for name, idx in zip(led_names, choice_idx_red.cpu().numpy()):
        print(f"{name}: column {idx}")
    #inspect_led_peaks(led_library, led_names, lambda_led)


if __name__ == "__main__":
    main()
