import os
import glob
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class H5ReflectanceDataset(Dataset):
    """
    Legge file .h5 da una cartella oppure da una lista di file.
    Cerca prima il dataset 'img\\', altrimenti 'spec'.

    Output per sample:
        cube : torch.Tensor [31, H, W]
    """

    def __init__(
        self,
        folder_path=None,
        file_list=None,
        dtype=torch.float32,
        crop_size=None,
        random_crop=False
    ):
        self.dtype = dtype
        self.crop_size = crop_size
        self.random_crop = random_crop

        if file_list is not None:
            self.files = sorted(file_list)
        elif folder_path is not None:
            self.files = sorted(glob.glob(os.path.join(folder_path, "*.h5")))
        else:
            raise ValueError("Devi fornire folder_path oppure file_list")

        if len(self.files) == 0:
            raise FileNotFoundError("Nessun file .h5 trovato")

    def __len__(self):
        return len(self.files)

    def _read_h5_cube(self, h5_path):
        with h5py.File(h5_path, "r") as f:
            if "img\\" in f:
                data = np.array(f["img\\"], dtype=np.float32)
            elif "spec" in f:
                data = np.array(f["spec"], dtype=np.float32)
            else:
                raise KeyError(f"Né 'img\\\\' né 'spec' trovati in {h5_path}")

        return data

    def _crop(self, cube):
        """
        cube: torch.Tensor [31, H, W]
        """
        if self.crop_size is None:
            return cube

        c, h, w = cube.shape
        cs = self.crop_size

        if h < cs or w < cs:
            raise ValueError(
                f"Crop size {cs} troppo grande per immagine di shape {tuple(cube.shape)}"
            )

        if self.random_crop:
            top = random.randint(0, h - cs)
            left = random.randint(0, w - cs)
        else:
            top = (h - cs) // 2
            left = (w - cs) // 2

        return cube[:, top:top + cs, left:left + cs]

    def __getitem__(self, idx):
        h5_path = self.files[idx]
        data = self._read_h5_cube(h5_path)

        cube = torch.tensor(data[:31, :, :], dtype=self.dtype)
        cube = self._crop(cube)

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
