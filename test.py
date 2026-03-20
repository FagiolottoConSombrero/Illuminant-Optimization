import torch
import matplotlib.pyplot as plt
from opt_network import *


def main():

    # --------------------------------------------------
    # paths
    # --------------------------------------------------
    checkpoint_path = "/Users/kolyszko/Scrivania/run_10_model_big/jointnet-epoch=57-val_loss=0.0774.ckpt"
    led_path = "/Users/kolyszko/Documents/MATLAB/Thouslite5.mat"
    camera_path = "/Users/kolyszko/Documents/NIKON-D810.csv"

    # --------------------------------------------------
    # load model
    # --------------------------------------------------
    model = JointNetwork.load_from_checkpoint(
        checkpoint_path,
        lr=1e-3,
        model_type=2,
        patience=50,
        n_ill=2,
        in_dim=6,  # 2 illuminanti -> 2 RGB -> 6 canali
        lambda_ang=0.2,
        led_path=led_path,
        camera_spd_path=camera_path
    )

    model.eval()

    # --------------------------------------------------
    # estrazione LED library
    # --------------------------------------------------
    led_library = model.ill_optimizer.led_library   # [15,20,301]

    _, logits = model.ill_optimizer(return_logits=True)                    # [2,15,20]

    # --------------------------------------------------
    # hard choice
    # --------------------------------------------------
    choice_idx = torch.argmax(logits, dim=-1)       # [2,15]

    print("Hard LED choices:")
    print(choice_idx)

    # --------------------------------------------------
    # costruzione illuminanti hard
    # --------------------------------------------------
    hard_illuminants = []

    for k in range(choice_idx.shape[0]):

        curves = []

        for led in range(15):

            idx = choice_idx[k, led]
            curves.append(led_library[led, idx])

        curves = torch.stack(curves)     # [15,301]

        illuminant = curves.sum(dim=0)   # [301]

        hard_illuminants.append(illuminant)

    hard_illuminants = torch.stack(hard_illuminants)   # [2,301]

    # --------------------------------------------------
    # plot illuminanti
    # --------------------------------------------------
    wl = np.linspace(400, 700, 301, dtype=np.float32)

    plt.figure(figsize=(10,5))

    for k in range(hard_illuminants.shape[0]):

        plt.plot(
            wl,
            hard_illuminants[k].detach().cpu().numpy(),
            linewidth=3,
            label=f"Optimized illuminant {k+1}"
        )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Power")
    plt.title("Optimized HARD illuminants best")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()