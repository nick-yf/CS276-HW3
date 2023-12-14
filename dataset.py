import os
import numpy as np
import torch
from torchvision import io
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, train_data_dir: str):
        self.train_layout_dir = os.path.join(train_data_dir, 'layout')
        self.train_mask_dir = os.path.join(train_data_dir, 'mask')
        self.train_layout_list = os.listdir(self.train_layout_dir)
        self.train_mask_list = os.listdir(self.train_mask_dir)
        self.train_layout_list.sort()
        self.train_mask_list.sort()
        assert len(self.train_layout_list) == len(self.train_mask_list)

    def __len__(self):
        return len(self.train_layout_list)

    def __getitem__(self, idx):
        layout = io.read_image(os.path.join(self.train_layout_dir, self.train_layout_list[idx]), mode=io.image.ImageReadMode.GRAY)
        mask = io.read_image(os.path.join(self.train_mask_dir, self.train_mask_list[idx]), mode=io.image.ImageReadMode.GRAY)
        layout = layout.float()
        mask = mask.float()
        return layout, mask


class TestDataset(Dataset):
    def __init__(self, test_data_dir: str):
        self.test_layout_dir = os.path.join(test_data_dir, 'layout')
        self.test_layout_list = os.listdir(self.test_layout_dir)
        self.test_layout_list.sort()

    def __len__(self):
        return len(self.test_layout_list)

    def __getitem__(self, idx):
        layout = io.read_image(os.path.join(self.test_layout_dir, self.test_layout_list[idx]), mode=io.image.ImageReadMode.GRAY)
        layout = layout.float()
        return layout
