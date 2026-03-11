import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class H5ReflectanceDataset(Dataset):
    """
    Legge file .h5 da una cartella.
    Cerca prima il dataset 'img', altrimenti 'spec'.

    Output per sample:
        cube : torch.Tensor [31, H, W]
        path : str
    """

    def __init__(self, folder_path, dtype=torch.float32):
        self.folder_path = folder_path
        self.dtype = dtype

        self.files = sorted(glob.glob(os.path.join(folder_path, "*.h5")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"Nessun file .h5 trovato in: {folder_path}")

    def __len__(self):
        return len(self.files)

    def _read_h5_cube(self, h5_path):
        with h5py.File(h5_path, "r") as f:
            # prova prima img, poi spec
            if "img\\" in f:
                data = np.array(f["img\\"], dtype=np.float32)
            elif "spec" in f:
                data = np.array(f["spec"], dtype=np.float32)

        return data

    def __getitem__(self, idx):
        h5_path = self.files[idx]
        data = self._read_h5_cube(h5_path)

        cube = torch.tensor(data[:31, :, :], dtype=self.dtype)
        return cube


def build_h5_dataloader(
    folder_path,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    dtype=torch.float32
):
    dataset = H5ReflectanceDataset(folder_path=folder_path, dtype=dtype)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return loader
