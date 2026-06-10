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


class H5PatchClassificationDataset(Dataset):
    """
    Dataset per classificazione da patch HSI aggregate.

    Expected H5:
        /x         -> N x C x H x W
                   oppure W x H x C x N se salvato da MATLAB
        /y         -> N, labels intere
        /source_id -> N, opzionale, usato per split senza leakage

    Output:
        x : torch.Tensor [C, H, W]
        y : torch.Tensor scalar long
    """

    def __init__(
        self,
        h5_path,
        indices=None,
        x_key="/x",
        y_key="/y",
        dtype=torch.float32
    ):
        self.h5_path = h5_path
        self.indices = None if indices is None else np.asarray(indices)
        self.x_key = x_key
        self.y_key = y_key
        self.dtype = dtype

        self.file = None
        self.x_ds = None
        self.y_ds = None

        with h5py.File(self.h5_path, "r") as f:
            if self.x_key not in f:
                raise KeyError(f"Dataset '{self.x_key}' non trovato nel file H5.")

            if self.y_key not in f:
                raise KeyError(f"Dataset '{self.y_key}' non trovato nel file H5.")

            self.x_shape = f[self.x_key].shape
            self.y_shape = f[self.y_key].shape

            if "/source_id" in f:
                n_samples = f["/source_id"].shape[0]
            else:
                n_samples = self._infer_n_samples_from_y_shape(self.y_shape)

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
                    f"x shape: {self.x_shape}, "
                    f"y shape: {self.y_shape}, "
                    f"n_samples: {n_samples}"
                )

        if self.indices is None:
            self.indices = np.arange(self.length_total)

    def _infer_n_samples_from_y_shape(self, y_shape):
        """
        Supporta:
            /y -> [N]
            /y -> [N, 1]
            /y -> [1, N]
        """
        if len(y_shape) == 1:
            return y_shape[0]

        if len(y_shape) == 2:
            if y_shape[0] == 1:
                return y_shape[1]
            if y_shape[1] == 1:
                return y_shape[0]
            return y_shape[0]

        raise ValueError(f"Shape label non supportata: {y_shape}")

    def __len__(self):
        return len(self.indices)

    def _open_file(self):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")
            self.x_ds = self.file[self.x_key]
            self.y_ds = self.file[self.y_key]

    def _read_label(self, real_idx):
        """
        Legge label da:
            /y shape [N]
            /y shape [N, 1]
            /y shape [1, N]
            /y shape [N, K] one-hot/probabilità
        """
        if len(self.y_shape) == 1:
            y = self.y_ds[real_idx]

        elif len(self.y_shape) == 2:
            # [N, 1]
            if self.y_shape[1] == 1:
                y = self.y_ds[real_idx, 0]

            # [1, N]
            elif self.y_shape[0] == 1:
                y = self.y_ds[0, real_idx]

            # [N, K] one-hot o probabilità
            else:
                y_vec = self.y_ds[real_idx]
                y = np.argmax(y_vec)

        else:
            raise ValueError(f"Shape label non supportata: {self.y_shape}")

        return int(y)

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

        y = self._read_label(real_idx)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

    def __getstate__(self):
        """
        Necessario per num_workers > 0.
        Evita di serializzare il file HDF5 aperto.
        """
        state = self.__dict__.copy()
        state["file"] = None
        state["x_ds"] = None
        state["y_ds"] = None
        return state

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
            self.x_ds = None
            self.y_ds = None



def split_by_source_id_classification(
    h5_path,
    train_ratio=0.70,
    val_ratio=0.30,
    test_ratio=0.00,
    seed=42
):
    """
    Split per classificazione a livello di source_id.

    Expected H5:
        /x         -> N x C x H x W
        /y         -> N
        /source_id -> N

    Returns:
        split = {
            "train_idx": ...,
            "val_idx": ...,
            "test_idx": ...,
            "train_sources": ...,
            "val_sources": ...,
            "test_sources": ...
        }

    Nota:
    - evita data leakage: patch della stessa immagine originale
      non finiscono in split diversi.
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio deve essere uguale a 1."

    rng = np.random.default_rng(seed)

    with h5py.File(h5_path, "r") as f:
        if "/source_id" not in f:
            raise KeyError(
                "Dataset '/source_id' non trovato. "
                "Serve per fare uno split corretto senza leakage."
            )

        if "/y" not in f:
            raise KeyError(
                "Dataset '/y' non trovato. "
                "Serve per la classificazione."
            )

        source_id = np.asarray(f["/source_id"]).squeeze()
        y = np.asarray(f["/y"]).squeeze()

    if source_id.ndim != 1:
        raise ValueError(f"/source_id deve essere 1D. Shape trovata: {source_id.shape}")

    if y.ndim != 1:
        raise ValueError(f"/y deve essere 1D con label intere. Shape trovata: {y.shape}")

    if len(source_id) != len(y):
        raise ValueError(
            f"/source_id e /y devono avere stessa lunghezza. "
            f"source_id: {len(source_id)}, y: {len(y)}"
        )

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