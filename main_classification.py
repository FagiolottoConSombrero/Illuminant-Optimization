import os
import argparse
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset_patch import *
from opt_network import *
from classification_models import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def infer_num_samples_from_y(y_shape):
    if len(y_shape) == 1:
        return y_shape[0]

    if len(y_shape) == 2:
        if y_shape[0] == 1:
            return y_shape[1]
        if y_shape[1] == 1:
            return y_shape[0]
        return y_shape[0]  # caso one-hot N x K

    raise ValueError(f"Shape label non supportata: {y_shape}")


def inspect_h5_classification_file(h5_path, x_key="/x", y_key="/y"):
    """
    Ricava:
        - numero bande input
        - numero classi
        - label uniche

    Supporta:
        /x -> N x C x H x W
        /x -> W x H x C x N
        /y -> N
        /y -> N x 1
        /y -> 1 x N
        /y -> N x K one-hot
    """

    with h5py.File(h5_path, "r") as f:
        if x_key not in f:
            raise KeyError(f"Dataset '{x_key}' non trovato.")

        if y_key not in f:
            raise KeyError(f"Dataset '{y_key}' non trovato.")

        x_shape = f[x_key].shape
        y = np.asarray(f[y_key])

        if "/source_id" in f:
            n_samples = f["/source_id"].shape[0]
        else:
            n_samples = infer_num_samples_from_y(y.shape)

    # infer labels
    if y.ndim == 1:
        labels = y

    elif y.ndim == 2:
        if y.shape[1] == 1:
            labels = y[:, 0]
        elif y.shape[0] == 1:
            labels = y[0, :]
        else:
            labels = np.argmax(y, axis=1)

    else:
        raise ValueError(f"Shape /y non supportata: {y.shape}")

    labels = labels.astype(np.int64)
    unique_labels = np.unique(labels)

    if unique_labels.min() != 0:
        raise ValueError(
            f"Le label devono partire da 0 per CrossEntropyLoss. "
            f"Trovate label: {unique_labels}. "
            f"Converti prima le label in 0, 1, ..., K-1."
        )

    num_classes = int(unique_labels.max() + 1)

    # infer input channels
    if len(x_shape) != 4:
        raise ValueError(f"Expected /x 4D, trovato shape: {x_shape}")

    if x_shape[0] == n_samples:
        layout = "NCHW"
        in_channels = x_shape[1]

    elif x_shape[-1] == n_samples:
        layout = "WHCN"
        in_channels = x_shape[2]

    else:
        raise ValueError(
            f"Shape /x non compatibile con n_samples. "
            f"x_shape={x_shape}, n_samples={n_samples}"
        )

    return {
        "x_shape": x_shape,
        "y_shape": y.shape,
        "layout": layout,
        "num_samples": n_samples,
        "in_channels": int(in_channels),
        "num_classes": num_classes,
        "unique_labels": unique_labels,
    }


parser = argparse.ArgumentParser("HSI textile classification", add_help=True)

# paths
parser.add_argument(
    "--data_path",
    required=True,
    type=str,
    help="Path al file H5 aggregato con /x, /y, /source_id"
)

parser.add_argument(
    "--checkpoint_path",
    default="./checkpoints_classification",
    type=str,
    help="Cartella in cui salvare i checkpoint"
)

parser.add_argument("--x_key", default="/x", type=str)
parser.add_argument("--y_key", default="/y", type=str)

# optimization
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--patience", default=30, type=int)
parser.add_argument("--epochs", default=200, type=int)

# split
parser.add_argument("--train_ratio", default=0.8, type=float)
parser.add_argument("--val_ratio", default=0.2, type=float)
parser.add_argument("--test_ratio", default=0.0, type=float)
parser.add_argument("--seed", default=42, type=int)

# model
parser.add_argument("--width", default=64, type=int)
parser.add_argument("--dropout", default=0.3, type=float)
parser.add_argument("--label_smoothing", default=0.0, type=float)

# dataloader
parser.add_argument("--num_workers", default=7, type=int)


def main():
    args = parser.parse_args()

    # --------------------------------------------------
    # 1. paths
    # --------------------------------------------------
    h5_path = args.data_path
    checkpoint_dir = args.checkpoint_path

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"File H5 non trovato: {h5_path}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # --------------------------------------------------
    # 2. reproducibility
    # --------------------------------------------------
    pl.seed_everything(args.seed, workers=True)

    # --------------------------------------------------
    # 3. inspect dataset
    # --------------------------------------------------
    info = inspect_h5_classification_file(
        h5_path=h5_path,
        x_key=args.x_key,
        y_key=args.y_key
    )

    print("Dataset mode: aggregated patch classification H5")
    print(f"H5 path:       {h5_path}")
    print(f"x shape:      {info['x_shape']}")
    print(f"y shape:      {info['y_shape']}")
    print(f"layout:       {info['layout']}")
    print(f"samples:      {info['num_samples']}")
    print(f"in_channels:  {info['in_channels']}")
    print(f"num_classes:  {info['num_classes']}")
    print(f"labels:       {info['unique_labels']}")

    in_channels = info["in_channels"]
    num_classes = info["num_classes"]

    # --------------------------------------------------
    # 4. split by source_id
    # --------------------------------------------------
    split = split_by_source_id_classification(
        h5_path=h5_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    train_dataset = H5PatchClassificationDataset(
        h5_path=h5_path,
        indices=split["train_idx"],
        x_key=args.x_key,
        y_key=args.y_key,
        dtype=torch.float32
    )

    val_dataset = H5PatchClassificationDataset(
        h5_path=h5_path,
        indices=split["val_idx"],
        x_key=args.x_key,
        y_key=args.y_key,
        dtype=torch.float32
    )

    print(f"Train source images: {len(split['train_sources'])}")
    print(f"Val source images:   {len(split['val_sources'])}")
    print(f"Train samples:       {len(train_dataset)}")
    print(f"Val samples:         {len(val_dataset)}")

    # --------------------------------------------------
    # 5. dataloaders
    # --------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0
    )

    # --------------------------------------------------
    # 6. model
    # --------------------------------------------------
    model = ClassificationNetwork(
        in_channels=in_channels,
        num_classes=num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        width=args.width,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing
    )

    # --------------------------------------------------
    # 7. callbacks
    # --------------------------------------------------
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="classifier-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval="epoch"
    )

    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=args.patience,
        verbose=True
    )

    # --------------------------------------------------
    # 8. trainer
    # --------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision=32,
        callbacks=[
            checkpoint_callback,
            lr_monitor,
            early_stop
        ],
        log_every_n_steps=10
    )

    # --------------------------------------------------
    # 9. train
    # --------------------------------------------------
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

    print("Training classificazione finito.")
    print("Best checkpoint:", checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()