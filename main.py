from torch.utils.data import random_split
from dataset import *
from opt_network import *
from utils import *


def main():
    # --------------------------------------------------
    # 1. paths
    # --------------------------------------------------
    data_dir = "/home/acp/datasets/SSD1/31bands_h5"
    led_path = "/home/acp/Documenti/Thouslite5.mat"
    camera_path = "/home/acp/Documenti/NIKON-D810.csv"
    checkpoint_dir = "./run_7_new_model_500_epochs"

    os.makedirs(checkpoint_dir, exist_ok=True)

    # --------------------------------------------------
    # 2. reproducibility
    # --------------------------------------------------
    pl.seed_everything(42, workers=True)

    # --------------------------------------------------
    # 3. dataset
    # --------------------------------------------------
    full_dataset = H5ReflectanceDataset(folder_path=data_dir,dtype=torch.float32)

    print(f"Numero file trovati: {len(full_dataset)}")

    # split train / val
    n_total = len(full_dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train

    train_dataset, val_dataset = random_split(full_dataset,
                                              [n_train, n_val],
                                              generator=torch.Generator().manual_seed(42))

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # --------------------------------------------------
    # 4. dataloaders
    # --------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=7,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=7,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )

    # --------------------------------------------------
    # 5. model
    # --------------------------------------------------
    model = JointNetwork(
        lr=1e-3,
        patience=50,
        model_type=1,
        n_ill=2,
        in_dim=6,                # 2 illuminanti -> 2 RGB -> 6 canali
        lambda_ang=0.2,
        led_path=led_path,
        camera_spd_path=camera_path
    )

    # --------------------------------------------------
    # 6. callbacks
    # --------------------------------------------------
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="jointnet-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=50,
        verbose=True
    )

    # --------------------------------------------------
    # 7. trainer
    # --------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=500,
        accelerator="auto",
        devices=1,
        precision=32,
        callbacks=[checkpoint_callback, lr_monitor, early_stop],
        log_every_n_steps=10
    )

    # --------------------------------------------------
    # 8. train
    # --------------------------------------------------
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    print("Training finito.")
    print("Best checkpoint:", checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()