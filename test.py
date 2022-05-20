import os
import PIL
import base64
import cv2
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser

from dataset import RealEstateDataset
from networks import StereoMagnificationModel
from loss import PSNR, SSIM
from utils import *

img_size = 224
batch_size = 1
checkpoint = 'debug/checkpoint.tar' 
data_dir = "real-estate-10k-run"
result_dir = 'no'
color_trans = torch.Tensor([65.738, 129.057, 25.064])

parser = ArgumentParser(description="Evaluation for MPIs")

parser.add_argument('--data_dir', default=data_dir, type=str, help="Directory of real-estate data")
parser.add_argument('--result_dir', default=result_dir, type=str, help="Directory of logging result")
parser.add_argument('--checkpoint', default=checkpoint, type=str, help="Directory of load checkpoint")
parser.add_argument('--img_size', default=img_size, type=int, help="training data image resolution")
parser.add_argument('--batch_size', default=batch_size, type=int, help="Mini batch size")
parser.add_argument('--color_trans', default=color_trans, type=int, help="RGB2YCbCr")

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = parser.parse_args()
    # load checkpoint
    ckt = torch.load(args.checkpoint)
    num_planes = ckt['num_planes']
    model = StereoMagnificationModel(num_mpi_planes=num_planes)
    model.load_state_dict(ckt['state_dict'])

    # Move to GPU, if available
    model = model.to(device)
    model.eval()
    color_trans = args.color_trans.to(device)

    # Custom dataloaders
    valid_dataset = RealEstateDataset(args.data_dir, img_size=args.img_size, is_valid=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    psnr = PSNR()
    ssim = SSIM(color_trans=color_trans)
    PSNRs = AverageMeter()
    SSIMs = AverageMeter()

    for i, (img, dep) in enumerate(valid_loader):
        # Move to GPU, if available
        img = img.type(torch.FloatTensor).to(device) 
        target = dep['tgt_img'].to(device)
        
        out = model(img)
        rgba_layers = mpi_from_net_output(out, dep)
        rel_pose = torch.matmul(dep['tgt_img_cfw'], dep['ref_img_wfc']).to(device)
        pred_image = mpi_render_view_torch(rgba_layers, rel_pose, dep['mpi_planes'][0], dep['intrinsics']).to(device)

        # Calculate loss
        _psnr = psnr(pred_image, target)
        _ssim = ssim(pred_image, target)
        
        PSNRs.update(_psnr.item())
        SSIMs.update(_ssim.item())
        
        print('{}/{} psnr : {:.4f}, ssim : {:.4f}'.format(i+1, len(valid_loader), _psnr.item(), _ssim.item()))
    
    print('epoch : {}, psnr : {:.4f}, ssim : {:.4f}'.format(ckt['epoch'], PSNRs.avg, SSIMs.avg))
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    with open(os.path.join(args.result_dir, 'result.txt'), 'a') as f:
        print('epoch : {}, psnr : {:.4f}, ssim : {:.4f}'.format(ckt['epoch'], PSNRs.avg, SSIMs.avg), file=f)

