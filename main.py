from sklearn.model_selection import train_test_split
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
    checkpoint_dir = "./run_12_random_crop"

    os.makedirs(checkpoint_dir, exist_ok=True)

    # --------------------------------------------------
    # 2. reproducibility
    # --------------------------------------------------
    pl.seed_everything(42, workers=True)

    # --------------------------------------------------
    # 3. dataset
    # --------------------------------------------------
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))

    if len(all_files) == 0:
        raise FileNotFoundError(f"Nessun file .h5 trovato in: {data_dir}")

    print(f"Numero file trovati: {len(all_files)}")

    # split train / val a livello di file
    train_files, val_files = train_test_split(
        all_files,
        test_size=0.1,
        random_state=42
    )

    train_dataset = H5ReflectanceDataset(
        file_list=train_files,
        dtype=torch.float32,
        crop_size=128,
        random_crop=True
    )

    val_dataset = H5ReflectanceDataset(
        file_list=val_files,
        dtype=torch.float32,
        crop_size=None,
        random_crop=False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # --------------------------------------------------
    # 4. dataloaders
    # --------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=7,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=7,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )

    # --------------------------------------------------
    # 5. model
    # --------------------------------------------------
    model = JointNetwork(
        lr=1e-4,
        patience=30,
        model_type=3,
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
        patience=65,
        verbose=True
    )

    # --------------------------------------------------
    # 7. trainer
    # --------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=1000,
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