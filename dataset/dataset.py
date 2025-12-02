import os
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
import h5py

class CtDataset(Dataset):

    def __init__(self, split, train_path=None, val_path=None,test_path=None, im_size=256, im_channels=1):
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        if split == 'train' and train_path:
            self.im_path = train_path
        elif split == 'val' and val_path:
            self.im_path = val_path
        elif split == 'test' and test_path:
            self.im_path = test_path
        else:
            raise ValueError("Invalid split. Provide train_path for 'train' or val_path for 'val'.")

        self.voltages, self.images = self.load_images(self.im_path)

    def load_images(self, im_path):
        assert os.path.isfile(im_path), f"File {im_path} does not exist"
        with h5py.File(im_path, 'r') as hdf5_file:
            voltages = np.array(hdf5_file['voltages']).astype(np.float32)
            images = np.array(hdf5_file['images']).astype(np.float32)
        print(f'Loaded {len(images)} images from {im_path}')
        return voltages, images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img = self.images[index].astype(np.float32)
        img_min, img_max = img.min(), img.max()
        img_range = img_max - img_min
        img_normalized = (img - img_min) / img_range
        im_tensor = torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(0)
        volt_tensor = torch.tensor(self.voltages[index], dtype=torch.float32)

        return volt_tensor,im_tensor
