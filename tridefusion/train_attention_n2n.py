import argparse
import json
import os
import random
import time
from pprint import pprint

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.amp import autocast, GradScaler

from utils.metrics import cal_psnr, cal_ssim, save_stats, save_samples
from utils.practices import OneCycleScheduler, adjust_learning_rate
from models.attention_unet import AttentionUNet  
from tridefusion.models.old.fluoro_msa_unet import MultiScaleAttentionUNet
from models.loss import Loss
from utils.data_loader import load_denoising_n2n_train, load_denoising_test_mix, fluore_to_tensor
from utils.misc import mkdirs, module_size
from ImageAnalysis.big_image_manager import BigImageManager


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description="Training Attention U-Net")
        self.add_argument('--exp-name', type=str, default='n2n', help='Experiment name')
        self.add_argument('--exp-dir', type=str, default='./experiments', help='Directory to save experiments')
        self.add_argument('--debug', action='store_true', help='Enable debug mode')
        self.add_argument('--data-root', type=str, default='./dataset', help='Root directory of the dataset')
        self.add_argument('--imsize', type=int, default=256, help='Input image size')
        self.add_argument('--chunk-size', type=int, default=128, help='Chunk size for processing large images')
        self.add_argument('--offset-size', type=int, default=64, help='Overlap size for chunks')
        self.add_argument('--in-channels', type=int, default=1, help='Number of input channels')
        self.add_argument('--out-channels', type=int, default=1, help='Number of output channels')
        self.add_argument('--transform', type=str, default='center_crop', choices=['center_crop', 'four_crop'],
                          help='Data augmentation type')
        self.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
        self.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
        self.add_argument('--loss-params', type=list, default=[1.0, 0.2, 0.1], help='Loss parameters for training')
        self.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        self.add_argument('--wd', type=float, default=0., help='Weight decay')
        self.add_argument('--cuda', type=int, default=0, help='CUDA device number')
        self.add_argument('--test-group', type=int, default=19)
        self.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
        self.add_argument('--noise-levels-train', type=list, default=[1, 2, 4, 8, 16])
        self.add_argument('--noise-levels-test', type=list, default=[1])
        self.add_argument('--teacher-checkpoint-path', type=str, default='./experiments/fluoro-msa/Dec_03_19_28/checkpoints/model_epoch150.pth')
        self.add_argument('--ckpt-freq', type=int, default=50, help='how many epochs to wait before saving model')
        self.add_argument('--print-freq', type=int, default=100, help='how many minibatches to wait before printing training status')
        self.add_argument('--log-freq', type=int, default=1, help='how many epochs to wait before logging training status')
        self.add_argument('--plot-epochs', type=int, default=5, help='how many epochs to wait before plotting test output')
        self.add_argument('--cmap', type=str, default='inferno', help='Colormap for output images')
        self.add_argument('--training-type', type=str, default='standard', choices=['standard', 'distillation'],
                          help='Type of training: "standard" or "distillation"')

    def parse(self):
        args = self.parse_args()
        date = time.strftime('%b_%d_%H_%M')
        args.run_dir = os.path.join(args.exp_dir, args.exp_name, date)
        args.ckpt_dir = os.path.join(args.run_dir, 'checkpoints')
        mkdirs([args.run_dir, args.ckpt_dir])
        
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        
        print(f"Random Seed: {args.seed}")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
        
        print('Arguments:')
        pprint(vars(args))
        with open(os.path.join(args.run_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        
        return args


def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True


def create_transform(transform_type, imsize):
    """Create data transformation based on type"""
    if transform_type == 'four_crop':
        transform = transforms.Compose([
            transforms.FiveCrop(imsize),
            transforms.Lambda(lambda crops: torch.stack([fluore_to_tensor(crop) for crop in crops[:4]])),
            transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
        ])
    else:
        transform = None
    return transform


def configure_optimizer_and_scheduler(model, lr, wd):
    """Configure optimizer and scheduler"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=[0.9, 0.99])
    scheduler = OneCycleScheduler(lr_max=lr, div_factor=10, pct_start=0.3)
    return optimizer, scheduler


def calc_loss(args, task_loss_fn, noisy_target, student_output, kd_loss_fn = None, teacher_output = None):
    reconstruction_loss = task_loss_fn(student_output, noisy_target)
    if args.training_type =='standard':
        return reconstruction_loss
    else:
        distillation_loss = kd_loss_fn(
                    F.log_softmax(student_output, dim=1),
                    F.softmax(teacher_output, dim=1)
                )
        total_loss = reconstruction_loss + 0.5 * distillation_loss
        return total_loss
    
#teacher_output = teacher_logits[batch_idx].to(device)
def get_loaders(args):

    load_denoising_n2n_train(args.data_root, batch_size=args.batch_size, noise_levels=args.noise_levels_train,
                            types=None, transform=transform, target_transform=transform,
                            patch_size=args.imsize, test_fov=args.test_group)
    train_loader = load_denoising_n2n_train(
        args.data_root,
        batch_size=args.batch_size,
        noise_levels=args.noise_levels_train,
        patch_size=args.imsize
    )
    test_loader = load_denoising_test_mix(
        args.data_root,
        batch_size=args.batch_size,
        noise_levels=args.noise_levels_test,
        patch_size=args.imsize
    )
    return train_loader, test_loader


def train_epoch(epoch, model, loss_fn, optimizer, train_loader, device, args, scheduler, logger):
    """Train model for one epoch"""
    model.train()
    train_psnr, train_loss, train_ssim = 0., 0., 0.
    scaler = GradScaler()
    
    for batch_idx, (noisy_input, noisy_target, clean) in enumerate(train_loader):
        noisy_input, noisy_target, clean = noisy_input.to(device), noisy_target.to(device), clean.to(device)
        if args.transform == 'four_crop':
            noisy_input, noisy_target, clean = [x.view(-1, *x.shape[2:]) for x in [noisy_input, noisy_target, clean]]
        with autocast():
            denoised = model(noisy_input)
            loss = loss_fn(denoised, noisy_target)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        adjust_learning_rate(optimizer, scheduler.step((epoch * len(train_loader) + batch_idx + 1) / (args.epochs * len(train_loader))))
        
        train_loss += loss.item()
        train_psnr += cal_psnr(clean, denoised).sum().item()
        train_ssim += cal_ssim(clean, denoised).sum().item()
        if (batch_idx + 1) % args.print_freq == 0:
            print(f'Epoch [{epoch}/{args.epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, PSNR: {train_psnr / len(train_loader.dataset):.4f}, SSIM: {train_ssim / len(train_loader.dataset):.4f}')
    
    logger['loss_task_train'].append(train_loss / len(train_loader))
    logger['psnr_train'].append(train_psnr / len(train_loader))
    logger['ssim_train'].append(train_ssim / len(train_loader))


def validate_epoch(epoch, model, loss_fn, test_loader, device, args, logger):
    test_psnr, test_loss, test_ssim = 0., 0., 0.
    with torch.no_grad():
        model.eval()
        for noisy, clean in test_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            denoised = model(noisy)
            loss = loss_fn(denoised, clean)
            test_loss += loss.item()
            test_psnr += cal_psnr(clean, denoised).sum().item()
            test_ssim += cal_ssim(clean, denoised).sum().item()
    logger['loss_task_train'].append(test_loss / len(test_loader))
    logger['psnr_train'].append(test_psnr / len(test_loader))

            psnr = psnr / n_test_samples
            mse = mse / n_test_pixels

def calculate_logits_for_teacher(teacher_model, dataloader, device, transform=None):
    """
    Calculate and store the logits of the teacher model for the entire dataset.
    
    Args:
        teacher_model (torch.nn.Module): Pre-trained teacher model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform calculations on (CPU or CUDA).
        transform (str, optional): Type of transformation applied to the input (e.g., 'four_crop').
    
    Returns:
        dict: A dictionary with keys corresponding to batch indices and values as tensors of logits.
    """
    teacher_model.eval()
    teacher_logits = {}
    with torch.no_grad():
        for batch_idx, (noisy_input, _, _) in enumerate(dataloader):
            noisy_input = noisy_input.to(device)
            if transform == 'four_crop':
                noisy_input = noisy_input.view(-1, *noisy_input.shape[2:])
            logits = teacher_model(noisy_input)
            teacher_logits[batch_idx] = logits.cpu()
    return teacher_logits

def initialize_params(args):
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    args.train_dir = args.run_dir + "/training"
    args.pred_dir = args.train_dir + "/predictions"
    mkdirs([args.train_dir, args.pred_dir])
    loss_task = Loss(*args.loss_params)
    logger = {}
    logger['psnr_train'] = []
    logger['loss_task_train'] = []
    logger['psnr_test'] = []
    logger['loss_task_test'] = []
    logger['ssim_train'] = []
    logger['mse_train'] = []
    logger['ssim_test'] = []
    logger['mse_test'] = []
    model = MultiScaleAttentionUNet(args.in_channels, args.out_channels).to(device)
    train_loader, test_loader = get_loaders(args)
    transform = create_transform(transform=args.transform, imsize=args.imsize)
    optimizer, scheduler = configure_optimizer_and_scheduler(model=model, lr=args.lr, wd=args.wd) 
    return logger, loss_task, model, optimizer, scheduler, train_loader, test_loader, transform

def train(args):
    print('Start training........................................................')
    logger, loss_task, model, optimizer, scheduler, train_loader, test_loader, transform = initialize_params(args)
    try:
        tic = time.time()
        for epoch in range(1, args.epochs + 1):
            train_epoch(model=model, loss_fn=loss_task, optimizer=optimizer, train_loader=train_loader, device=args.device, args=args, scheduler=scheduler, logger=logger)
            if epoch % args.ckpt_freq == 0:
                ckpt_path = os.path.join(args.ckpt_dir, f"student_model_epoch{epoch}.pth")
                torch.save(model.state_dict(), ckpt_path)
            if epoch % args.plot_epochs == 0:
                fixed_denoised = model(fixed_test_noisy)
                samples = torch.cat((fixed_test_noisy[:4], fixed_denoised[:4], fixed_test_clean[:4],
                                     fixed_denoised[:4] - fixed_test_clean[:4]))
                save_samples(args.pred_dir, samples, epoch, 'fixed_test', epoch=True, cmap=args.cmap)


        tic2 = time.time()
        print("Finished training {} epochs using {} seconds"
              .format(args.epochs, tic2 - tic))
        args.training_time = tic2 - tic
        args.n_params, args.n_layers = module_size(model)
        x_axis = np.arange(args.log_freq, args.epochs + 1, args.log_freq)
        save_stats(args.train_dir, logger, x_axis, ['psnr_train', 'psnr_test'])
        save_stats(args.train_dir, logger, x_axis, ['loss_task_train', 'loss_task_test'])
        save_stats(args.train_dir, logger, x_axis, ['ssim_task_train', 'ssim_task_test'])
        with open(args.run_dir + "/args.txt", 'w') as args_file:
            json.dump(vars(args), args_file, indent=4)
        print("Training completed successfully.")
    except KeyboardInterrupt:
        print('Keyboard Interrupt captured...Saving models & training logs')
        tic2 = time.time()
        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "/model_epoch{}.pth".format(epoch)))
def train_with_distillation(args):
    pass