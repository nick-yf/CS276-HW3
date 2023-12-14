import argparse
import os
import json
import math
import time
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import io

from dataset import TrainDataset, TestDataset
from model import EGAN_G, EGAN_D
from lithosim import lithosim_cuda as litho


def save_model_and_result(
    output_dir: str,
    epoch: int,
    model_G: nn.Module,
    model_D: nn.Module,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
    step: int,
    test_dataloader: DataLoader,
    device: torch.device,
    writer: SummaryWriter,
    threshold: float,
    kernels: torch.Tensor,
    weight: torch.Tensor,
    kernels_def: torch.Tensor,
    weight_def: torch.Tensor,
    kernel_num: int
):
    save_dir = os.path.join(output_dir, f'epoch_{epoch}')
    downsample = nn.AvgPool2d(8, stride=8)
    upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(
        {
            "model_G": model_G.state_dict(),
            "model_D": model_D.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D": optimizer_D.state_dict(),
            "step": step
        },
        os.path.join(save_dir, 'model.pt')
    )
    with torch.no_grad():
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
            L2_error = (wafer_nom - layout).abs().sum()
            PVB = (wafer_min - wafer_max).abs().sum()
            torchvision.utils.save_image(layout, os.path.join(save_dir, f"layout_{i}.png"))
            torchvision.utils.save_image(wafer_nom, os.path.join(save_dir, f"wafer_{i}.png"))
            torchvision.utils.save_image(fake_mask, os.path.join(save_dir, f"mask_{i}.png"))
            print(f"time cost: {(end_time - start_time) * 100}, L2 error: {L2_error.item()}, PVB: {PVB.item()}")
            cost += (end_time - start_time) * 100 + PVB.item() + L2_error.item()
        print('Cost: {}'.format(cost))
        writer.add_scalar('cost', cost, epoch)


def main():
    parser = argparse.ArgumentParser(description='take parameters')
    parser.add_argument("train_data_dir", type=str, help="path of the directory to training data")
    parser.add_argument("test_data_dir", type=str, help="path of the directory to test data")
    parser.add_argument("output_dir", type=str, help="path of the directory to output result")
    # training parameters
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches when training")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--cpu_num", type=int, default=8, help="number of cpu threads to use when loading data")
    parser.add_argument("--log_steps", type=int, default=100, help="number of steps when print log")
    parser.add_argument("--resume_epoch", type=int, default=None, help="the epoch to load model")
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

    writer = SummaryWriter(os.path.join(args.output_dir, 'log'))

    train_data_dir = os.path.abspath(args.train_data_dir)
    test_data_dir = os.path.abspath(args.test_data_dir)
    output_dir = os.path.abspath(args.output_dir)
    train_dataset = TrainDataset(train_data_dir)
    test_dataset = TestDataset(test_data_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.cpu_num)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    model_G = EGAN_G()
    model_D = EGAN_D()
    downsample = nn.AvgPool2d(8, stride=8)
    adversial_loss = nn.BCEWithLogitsLoss()
    pixel_loss = nn.MSELoss()

    model_G = model_G.to(device)
    model_D = model_D.to(device)
    adversial_loss = adversial_loss.to(device)
    pixel_loss = pixel_loss.to(device)

    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=args.lr)

    if args.resume_epoch is not None:
        checkpoint = torch.load(os.path.join(args.load_dir, 'model.pt'))
        model_G.load_state_dict(checkpoint['model_G'])
        model_D.load_state_dict(checkpoint['model_D'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        step = checkpoint['step']
    else:
        step = 0
        model_D.eval()
        model_G.train()
        save_model_and_result(
            output_dir,
            0,
            model_G,
            model_D,
            optimizer_G,
            optimizer_D,
            step,
            test_dataloader,
            device,
            writer,
            threshold,
            kernels,
            weight,
            kernels_def,
            weight_def,
            args.kernel_num
        )

    # # ILT pretrain
    # for i, batch in enumerate(train_dataloader):
    #     layout, _ = batch
    #     layout = layout.to(device)
    #     while True:
    #         optimizer_G.zero_grad()
    #         fake_mask = model_G(layout)
    #         fake_mask.retain_grad()
    #         wafer, _ = litho.lithosim(fake_mask, threshold, kernels, weight, None, save_bin_wafer_image=False, kernels_number=args.kernel_num, dose=1.0)
    #         L2_error = pixel_loss(wafer, layout)
    #         print(L2_error)
    #         if L2_error < 5e-4:
    #             break
    #         L2_error.backward()
    #         optimizer_G.step()
    #         writer.add_scalar('L2_loss', L2_error.item(), i)

    for e in range(args.epochs):
        model_D.train()
        model_G.train()
        for i, batch in enumerate(train_dataloader):
            layout, mask = batch
            layout = layout.to(device)
            mask = mask.to(device)
            layout = downsample(layout)
            mask = downsample(mask)
            real_label = torch.ones((layout.shape[0], 1)).to(device)
            fake_label = torch.zeros((layout.shape[0], 1)).to(device)
            fake_mask = model_G(layout)
            real_pair = torch.cat((mask, layout), dim=1)
            fake_pair = torch.cat((fake_mask, layout), dim=1)
            # Train G
            optimizer_G.zero_grad()
            fake_output = model_D(fake_pair)
            G_loss = adversial_loss(fake_output, real_label) + args.alpha * pixel_loss(fake_mask, mask)
            G_loss.backward()
            optimizer_G.step()
            # Train D
            optimizer_D.zero_grad()
            real_output = model_D(real_pair)
            fake_output = model_D(fake_pair.detach())
            D_loss = adversial_loss(real_output, real_label) + adversial_loss(fake_output, fake_label)
            D_loss.backward()
            optimizer_D.step()
            # ILT refinement
            # while True:
            #     optimizer_G.zero_grad()
            #     fake_mask = model_G(layout)
            #     fake_mask.retain_grad()
            #     wafer, _ = litho.lithosim(fake_mask, threshold, kernels, weight, None, save_bin_wafer_image=False, kernels_number=args.kernel_num, dose=1.0)
            #     L2_error = pixel_loss(wafer, layout)
            #     L2_error.backward()
            #     if torch.mean(torch.abs(fake_mask.grad)) < 5e-4:
            #         break
            #     optimizer_G.step()
            # print log
            writer.add_scalar('D_loss', D_loss.item(), step)
            writer.add_scalar('G_loss', G_loss.item(), step)
            # save result
            if step % args.log_steps == 0:
                print(f'Epoch: {e}, Step: {i}, D_loss: {D_loss.item():.5f}, G_loss: {G_loss.item():.5f}')
                with torch.no_grad():
                    n = int(math.ceil(layout.shape[0] ** 0.5))
                    _, wafer_nom = litho.lithosim(fake_mask, threshold, kernels, weight, None, save_bin_wafer_image=False, kernels_number=args.kernel_num, dose=1.0)
                    generated_wafers = torch.ones((n * 256, n * 256))
                    original_wafers = torch.ones((n * 256, n * 256))
                    for j in range(layout.shape[0]):
                        generated_wafers[(j // n) * 256:((j // n) + 1) * 256, (j % n) * 256:((j % n) + 1) * 256] = wafer_nom[j, :, :]
                        original_wafers[(j // n) * 256:((j // n) + 1) * 256, (j % n) * 256:((j % n) + 1) * 256] = layout[j, :, :]
                    generated_wafers = generated_wafers * 255
                    generated_wafers = generated_wafers.cpu().to(torch.uint8)
                    original_wafers = original_wafers * 255
                    original_wafers = original_wafers.cpu().to(torch.uint8)
                    writer.add_image('layout', original_wafers, step, dataformats='HW')
                    writer.add_image('generated_wafer', generated_wafers, step, dataformats='HW')
            step += 1
        print('Epoch {} finished'.format(e))
        # save model
        model_D.eval()
        model_G.eval()
        save_model_and_result(
            output_dir,
            e+1,
            model_G,
            model_D,
            optimizer_G,
            optimizer_D,
            step,
            test_dataloader,
            device,
            writer,
            threshold,
            kernels,
            weight,
            kernels_def,
            weight_def,
            args.kernel_num
        )

if __name__ == '__main__':
    main()
