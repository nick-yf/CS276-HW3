import argparse
import os
import json
import math
import time
import shutil

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import io

from dataset import TrainDataset, TestDataset
from model import EGAN_G, EGAN_D
from lithosim import lithosim_cuda as litho


def evaluate_model(
    model_G: nn.Module,
    model_D: nn.Module,
    test_dataloader: DataLoader,
    device: torch.device,
    threshold: float,
    kernels: torch.Tensor,
    weight: torch.Tensor,
    kernels_def: torch.Tensor,
    weight_def: torch.Tensor,
    kernel_num: int
):
    downsample = nn.AvgPool2d(8, stride=8)
    upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
    model_G.eval()
    model_D.eval()
    with torch.no_grad():
        L2_error = 0
        PVB = 0
        cost = 0
        for i, layout in enumerate(test_dataloader):
            layout = layout.to(device)
            start_time = time.time()
            layout_downsampled = downsample(layout)
            fake_mask = model_G(layout_downsampled)
            fake_mask = upsample(fake_mask)
            end_time = time.time()
            _, wafer_nom = litho.lithosim(fake_mask, threshold, kernels, weight, None, save_bin_wafer_image=False, kernels_number=kernel_num, dose=1.0)
            _, wafer_min = litho.lithosim(fake_mask, threshold, kernels_def, weight_def, None, save_bin_wafer_image=False, kernels_number=kernel_num, dose=0.98)
            _, wafer_max = litho.lithosim(fake_mask, threshold, kernels, weight, None, save_bin_wafer_image=False, kernels_number=kernel_num, dose=1.02)
            L2_error += (wafer_nom - layout).abs().sum()
            PVB += (wafer_min - wafer_max).abs().sum()
            print(f"time cost: {(end_time - start_time) * 100}, L2 error: {(wafer_nom - layout).abs().sum().item()}, PVB: {(wafer_min - wafer_max).abs().sum().item()}")
            cost += (end_time - start_time) * 100 + (wafer_min - wafer_max).abs().sum().item() + (wafer_nom - layout).abs().sum().item()
        cost /= len(test_dataloader)
        L2_error /= len(test_dataloader)
        PVB /= len(test_dataloader)
        print(f'Average cost: {cost}, Average L2 error: {L2_error}, Average PVB: {PVB}')


def main():
    parser = argparse.ArgumentParser(description='take parameters')
    parser.add_argument("test_data_dir", type=str, help="path of the directory to test data")
    parser.add_argument("ckpt_path", type=str, default=None, help="the path of checkpoint")
    # lithosim parameters
    parser.add_argument('--kernel_data_path', type=str, default='lithosim/lithosim_kernels/torch_tensor')
    parser.add_argument('--kernel_num', type=int, default=24, help='24 SOCS kernels')
    parser.add_argument('--alpha', type=float, default=50, help='')
    parser.add_argument('--beta', type=int, default=4, help='')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
        print("CUDA not available, use CPU instead")

    kernel_torch_data_path = 'lithosim/lithosim_kernels/torch_tensor'
    kernels_path = os.path.join(kernel_torch_data_path, 'kernel_focus_tensor.pt')
    weight_path = os.path.join(kernel_torch_data_path, 'weight_focus_tensor.pt')
    kernels_def_path = os.path.join(kernel_torch_data_path, 'kernel_defocus_tensor.pt')
    weight_def_path = os.path.join(kernel_torch_data_path, 'weight_defocus_tensor.pt')
    kernels = torch.load(kernels_path, map_location=device)
    weight = torch.load(weight_path, map_location=device)
    kernels_def = torch.load(kernels_def_path, map_location=device)
    weight_def = torch.load(weight_def_path, map_location=device)
    threshold = 0.225

    test_data_dir = os.path.abspath(args.test_data_dir)
    test_dataset = TestDataset(test_data_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model_G = EGAN_G()
    model_D = EGAN_D()

    model_G = model_G.to(device)
    model_D = model_D.to(device)
    checkpoint = torch.load(args.ckpt_path)
    model_G.load_state_dict(checkpoint['model_G'])
    model_D.load_state_dict(checkpoint['model_D'])

    evaluate_model(
        model_G,
        model_D,
        test_dataloader,
        device,
        threshold,
        kernels,
        weight,
        kernels_def,
        weight_def,
        args.kernel_num
    )


if __name__ == '__main__':
    main()
