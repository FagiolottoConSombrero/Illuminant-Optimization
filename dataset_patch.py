import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class H5PatchReconstructionDataset(Dataset):
    """
    Dataset per ricostruzione spettrale da patch HSI aggregate.

    Expected H5:
        /x         -> N x C x H x W
        /source_id -> N   opzionale, usato solo per split

    Output:
        x : torch.Tensor [C, H, W]
    """

    def __init__(
        self,
        h5_path,
        indices=None,
        dtype=torch.float32
    ):
        self.h5_path = h5_path
        self.indices = None if indices is None else np.asarray(indices)
        self.dtype = dtype

        self.file = None
        self.x_ds = None

        with h5py.File(self.h5_path, "r") as f:
            if "/x" not in f:
                raise KeyError("Dataset '/x' non trovato nel file H5.")

            self.x_shape = f["/x"].shape

            if "/source_id" in f:
                n_samples = f["/source_id"].shape[0]
            else:
                n_samples = self.x_shape[0]

            # Caso ideale: /x letto come N x C x H x W
            if self.x_shape[0] == n_samples:
                self.layout = "NCHW"
                self.length_total = self.x_shape[0]

            # Caso MATLAB/HDF5: /x letto come W x H x C x N
            elif self.x_shape[-1] == n_samples:
                self.layout = "WHCN"
                self.length_total = self.x_shape[-1]

            else:
                raise ValueError(
                    f"Shape /x non compatibile. "
                    f"x shape: {self.x_shape}, n_samples: {n_samples}"
                )

        if self.indices is None:
            self.indices = np.arange(self.length_total)

    def __len__(self):
        return len(self.indices)

    def _open_file(self):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")
            self.x_ds = self.file["/x"]

    def __getitem__(self, idx):
        self._open_file()

        real_idx = int(self.indices[idx])

        if self.layout == "NCHW":
            x = self.x_ds[real_idx]            # C x H x W

        elif self.layout == "WHCN":
            x = self.x_ds[:, :, :, real_idx]   # W x H x C
            x = np.transpose(x, (2, 1, 0))     # C x H x W

        x = np.ascontiguousarray(x)
        x = torch.tensor(x, dtype=self.dtype)

        return x

    def __getstate__(self):
        """
        Necessario per num_workers > 0.
        Evita di serializzare il file HDF5 aperto.
        """
        state = self.__dict__.copy()
        state["file"] = None
        state["x_ds"] = None
        return state

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
            self.x_ds = None


def split_by_source_id_reconstruction(
    h5_path,
    train_ratio=0.70,
    val_ratio=0.30,
    test_ratio=0.00,
    seed=42
):
    """
    Split a livello di immagine originale.

    Tutte le patch con lo stesso source_id finiscono nello stesso split.
    Non usa y.
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = np.random.default_rng(seed)

    with h5py.File(h5_path, "r") as f:
        if "/source_id" not in f:
            raise KeyError(
                "Dataset '/source_id' non trovato. "
                "Serve per fare uno split corretto senza leakage."
            )

        source_id = np.asarray(f["/source_id"])

    unique_sources = np.unique(source_id)
    rng.shuffle(unique_sources)

    n_sources = len(unique_sources)

    n_train = int(round(train_ratio * n_sources))
    n_val = int(round(val_ratio * n_sources))

    if n_train + n_val > n_sources:
        n_val = n_sources - n_train

    train_sources = unique_sources[:n_train]
    val_sources = unique_sources[n_train:n_train + n_val]
    test_sources = unique_sources[n_train + n_val:]

    train_idx = np.where(np.isin(source_id, train_sources))[0]
    val_idx = np.where(np.isin(source_id, val_sources))[0]
    test_idx = np.where(np.isin(source_id, test_sources))[0]

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    split = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "train_sources": train_sources,
        "val_sources": val_sources,
        "test_sources": test_sources,
    }

    return split


def build_h5_patch_reconstruction_dataloaders(
    h5_path,
    batch_size=16,
    train_ratio=0.70,
    val_ratio=0.30,
    test_ratio=0.00,
    seed=42,
    num_workers=4,
    pin_memory=True,
    dtype=torch.float32
):
    split = split_by_source_id_reconstruction(
        h5_path=h5_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    train_dataset = H5PatchReconstructionDataset(
        h5_path=h5_path,
        indices=split["train_idx"],
        dtype=dtype
    )

    val_dataset = H5PatchReconstructionDataset(
        h5_path=h5_path,
        indices=split["val_idx"],
        dtype=dtype
    )

    test_dataset = H5PatchReconstructionDataset(
        h5_path=h5_path,
        indices=split["test_idx"],
        dtype=dtype
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    return train_loader, val_loader, test_loader, split