from sklearn.model_selection import train_test_split
from dataset import *
from opt_network import *
from utils import *
from dataset_patch import *
import argparse


parser = argparse.ArgumentParser('Set args', add_help=False)

# paths
parser.add_argument("--data_path", default="/home/acp/datasets/SSD1/31bands_h5", type=str, help="img data path")
parser.add_argument("--led_path", default="/home/acp/Documenti/Thouslite5.mat", type=str, help="lightbooth data path")
parser.add_argument("--camera_path", default="/home/acp/Documenti/NIKON-D810.csv", type=str, help="rgb sensor camera data path")
parser.add_argument("--checkpoint_path", default="", type=str, help="save path")
parser.add_argument("--patches", default=False, type=bool, help="True: read aggregated patch H5; False: read folder of H5 files")

# optimization
parser.add_argument("--lr", default=1e-4, type=float, help="base learning rate")
parser.add_argument("--batch_size", default=4, type=int, help="batch size")
parser.add_argument("--model", default=1, type=int, help="1: mlp ; 2: CNN ; 3: MST++")
parser.add_argument("--patience", default=30, type=int, help="patience for early stopping")
parser.add_argument("--epochs", default=1000, type=int, help="training epochs")
parser.add_argument("--only_rendering", default=False, type=bool, help="True: optimize illuminant only")




def main():
    args = parser.parse_args()
    # --------------------------------------------------
    # 1. paths
    # --------------------------------------------------
    data_dir = args.data_path
    led_path = args.led_path
    camera_path = args.camera_path
    checkpoint_dir = args.checkpoint_path

    os.makedirs(checkpoint_dir, exist_ok=True)

    # --------------------------------------------------
    # 2. reproducibility
    # --------------------------------------------------
    pl.seed_everything(42, workers=True)

      # --------------------------------------------------
    # 3. dataset
    # --------------------------------------------------

    if args.patches is False:
        # ==================================================
        # Original dataset: folder with many .h5 files
        # ==================================================

        all_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))

        if len(all_files) == 0:
            raise FileNotFoundError(f"Nessun file .h5 trovato in: {data_dir}")

        print("Dataset mode: original H5 files")
        print(f"Numero file trovati: {len(all_files)}")

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

        train_batch_size = args.batch_size
        val_batch_size = 1

    else:
        # ==================================================
        # Patch dataset: single aggregated H5
        # Expected:
        #   /x         -> N x 31 x H x W
        #   /source_id -> N
        # ==================================================

        if not os.path.isfile(data_dir):
            raise FileNotFoundError(
                f"Con --patches True, data_path deve essere un file H5. Ricevuto: {data_dir}"
            )

        print("Dataset mode: aggregated patches H5")
        print(f"Patch dataset: {data_dir}")

        split = split_by_source_id_reconstruction(
            h5_path=data_dir,
            train_ratio=0.9,
            val_ratio=0.1,
            test_ratio=0.0,
            seed=42
        )

        train_dataset = H5PatchReconstructionDataset(
            h5_path=data_dir,
            indices=split["train_idx"],
            dtype=torch.float32
        )

        val_dataset = H5PatchReconstructionDataset(
            h5_path=data_dir,
            indices=split["val_idx"],
            dtype=torch.float32
        )

        train_batch_size = args.batch_size
        val_batch_size = args.batch_size

        print(f"Train source images: {len(split['train_sources'])}")
        print(f"Val source images:   {len(split['val_sources'])}")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # --------------------------------------------------
    # 4. dataloaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=7,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=7,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )

    # --------------------------------------------------
    # 5. model
    # --------------------------------------------------
    if args.only_rendering:
        model = IllNetwork(
            lr=args.lr,
            patience=args.patience,
            model_type=args.model,
            n_ill=2,               # 2 illuminanti -> 2 RGB -> 6 canali
            led_path=led_path,
            camera_spd_path=camera_path
        )

    else:
        model = JointNetwork(
            lr=args.lr,
            patience=args.patience,
            model_type=args.model,
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
        max_epochs=args.epochs,
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