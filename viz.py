import torch
import matplotlib.pyplot as plt
import numpy as np

from opt_network import *
from dataset import *
from utils import *


def build_hard_illuminants(model):
    """
    Costruisce gli illuminanti hard a partire dai logits
    """
    led_library = model.ill_optimizer.led_library  # [15,20,L]

    # logits dal modello (MLP version)
    _, logits = model.ill_optimizer(return_logits=True)

    # scelta hard
    choice_idx = torch.argmax(logits, dim=-1)  # [K,15]

    hard_illuminants = []

    for k in range(choice_idx.shape[0]):
        curves = []

        for led in range(15):
            idx = choice_idx[k, led]
            curves.append(led_library[led, idx])

        curves = torch.stack(curves)       # [15,L]
        illuminant = curves.sum(dim=0)     # [L]

        hard_illuminants.append(illuminant)

    return torch.stack(hard_illuminants)   # [K,L]


def main():

    # --------------------------------------------------
    # paths
    # --------------------------------------------------
    checkpoint_path = "/Users/kolyszko/Scrivania/run_9_only_ill/jointnet-epoch=100-val_loss=0.0269.ckpt"
    led_path = "/Users/kolyszko/Documents/MATLAB/Thouslite5.mat"
    camera_path = "/Users/kolyszko/Documents/NIKON-D810.csv"
    data_dir = "/Volumes/Lexar/31bands_h5"


    # --------------------------------------------------
    # load model
    # --------------------------------------------------
    model = JointNetwork.load_from_checkpoint(
        checkpoint_path,
        lr=1e-3,
        patience=50,
        model_type=2,
        n_ill=2,
        in_dim=6,
        lambda_ang=0.2,
        led_path=led_path,
        camera_spd_path=camera_path
    )

    model.eval()

    # --------------------------------------------------
    # dataset (prendo un sample)
    # --------------------------------------------------
    dataset = H5ReflectanceDataset(folder_path=data_dir, dtype=torch.float32)
    ref = dataset[0].unsqueeze(0)   # [1,31,H,W]

    # --------------------------------------------------
    # hard illuminants
    # --------------------------------------------------
    hard_ills = build_hard_illuminants(model)  # [2,L]

    # --------------------------------------------------
    # rendering RGB
    # --------------------------------------------------
    rgb1, rgb2 = render_rgb(ref, hard_ills, camera_path)

    # [1,3,H,W] -> [H,W,3]
    rgb1 = rgb1[0].permute(1, 2, 0).detach().cpu().numpy()
    rgb2 = rgb2[0].permute(1, 2, 0).detach().cpu().numpy()

    # clamp per sicurezza
    rgb1 = np.clip(rgb1, 0, 1)
    rgb2 = np.clip(rgb2, 0, 1)
    print("rgb1 min/max:", rgb1.min(), rgb1.max(), rgb1.mean())
    print("rgb2 min/max:", rgb2.min(), rgb2.max(), rgb2.mean())

    # --------------------------------------------------
    # plot
    # --------------------------------------------------
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(rgb1)
    plt.title("Illuminant 1 (hard)")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(rgb2)
    plt.title("Illuminant 2 (hard)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()