import os
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

def ConvLayer(ni, nf, ks=3, stride=1, padding=None, bias=None, norm_type='bn', transpose=False, dilation=1, groups=1, act_cls=nn.ReLU):
    if padding is None:
        padding = (ks - 1) // 2
    conv_layer = [] # Conv, norm, activation

    # Convlayer
    if transpose:
        conv_layer.append(nn.ConvTranspose2d(in_channels=ni, 
                                             out_channels=nf, 
                                             kernel_size=ks, 
                                             stride=stride, 
                                             padding=padding, 
                                             bias=bias, 
                                             dilation=dilation, 
                                             groups=groups))
    else:
        conv_layer.append(nn.Conv2d(in_channels=ni, 
                                    out_channels=nf, 
                                    kernel_size=ks, 
                                    stride=stride, 
                                    padding=padding, 
                                    bias=bias, 
                                    dilation=dilation, 
                                    groups=groups))

    # Norm layer
    if norm_type == 'bn':
        conv_layer.append(nn.BatchNorm2d(nf))
    elif norm_type == 'InstantNorm':
        conv_layer.append(nn.InstanceNorm2d(nf))

    # activation layer
    if act_cls is not None:
        conv_layer.append(act_cls())

    return nn.Sequential(*conv_layer)

class StereoMagnificationModel(nn.Module):

  def __init__(self, num_mpi_planes):
    super(StereoMagnificationModel, self).__init__()
    ngf = 3 + num_mpi_planes * 3
    nout = 3 + num_mpi_planes * 2
    self.ngf = ngf
    self.nout = nout
    InstanceNorm = 'InstantNorm'
    self.cnv1_1 = ConvLayer(ngf,ngf, ks=3, stride=1, norm_type=InstanceNorm)                                  # 224
    self.cnv1_2 = ConvLayer(ngf,ngf*2, ks=3, stride=2, norm_type=InstanceNorm)                                # 112
    
    self.cnv2_1 = ConvLayer(ngf*2,ngf*2, ks=3, stride=1, norm_type=InstanceNorm)                              # 112
    self.cnv2_2 = ConvLayer(ngf*2,ngf*4, ks=3, stride=2, norm_type=InstanceNorm)                              # 56
    
    self.cnv3_1 = ConvLayer(ngf*4,ngf*4, ks=3, stride=1, norm_type=InstanceNorm)                              # 56
    self.cnv3_2 = ConvLayer(ngf*4,ngf*4, ks=3, stride=1, norm_type=InstanceNorm)                              # 56
    self.cnv3_3 = ConvLayer(ngf*4,ngf*8, ks=3, stride=2, norm_type=InstanceNorm)                              # 28
    
    self.cnv4_1 = ConvLayer(ngf*8,ngf*8, ks=3, stride=1, dilation=2, padding=2, norm_type=InstanceNorm)       # 28
    self.cnv4_2 = ConvLayer(ngf*8,ngf*8, ks=3, stride=1, dilation=2, padding=2, norm_type=InstanceNorm)       # 28
    self.cnv4_3 = ConvLayer(ngf*8,ngf*8, ks=3, stride=1, dilation=2, padding=2, norm_type=InstanceNorm)       # 28
    
    self.cnv5_1 = ConvLayer(ngf*16,ngf*4, ks=4, stride=2, transpose=True, padding=1, norm_type=InstanceNorm)  # 56
    self.cnv5_2 = ConvLayer(ngf*4,ngf*4, ks=3, stride=1, norm_type=InstanceNorm)                              # 56
    self.cnv5_3 = ConvLayer(ngf*4,ngf*4, ks=3, stride=1, norm_type=InstanceNorm)                              # 56
    
    self.cnv6_1 = ConvLayer(ngf*8,ngf*2, ks=4, stride=2, transpose=True, padding=1, norm_type=InstanceNorm)   # 112
    self.cnv6_2 = ConvLayer(ngf*2,ngf*2, ks=3, stride=1, norm_type=InstanceNorm)                              # 112
    
    self.cnv7_1 = ConvLayer(ngf*4,nout, ks=4, stride=2, transpose=True, padding=1, norm_type=InstanceNorm)    # 224
    self.cnv7_2 = ConvLayer(nout,nout, ks=3, stride=1, norm_type=InstanceNorm)                                # 224
    
    self.cnv8_1 = ConvLayer(nout,nout, ks=1, stride=1, norm_type=None, act_cls=nn.Tanh)                       # 224
  
  def forward(self, x):
    out_cnv1_1 = self.cnv1_1(x)
    out_cnv1_2 = self.cnv1_2(out_cnv1_1)
    
    out_cnv2_1 = self.cnv2_1(out_cnv1_2)
    out_cnv2_2 = self.cnv2_2(out_cnv2_1)
    
    out_cnv3_1 = self.cnv3_1(out_cnv2_2)
    out_cnv3_2 = self.cnv3_2(out_cnv3_1)
    out_cnv3_3 = self.cnv3_3(out_cnv3_2)
    
    out_cnv4_1 = self.cnv4_1(out_cnv3_3)
    out_cnv4_2 = self.cnv4_2(out_cnv4_1)
    out_cnv4_3 = self.cnv4_3(out_cnv4_2)
    
    # add skip connection
    in_cnv5_1 = torch.cat([out_cnv4_3, out_cnv3_3],1)
    
    out_cnv5_1 = self.cnv5_1(in_cnv5_1)
    out_cnv5_2 = self.cnv5_2(out_cnv5_1)
    out_cnv5_3 = self.cnv5_3(out_cnv5_2)
    
    # add skip connection
    in_cnv6_1 = torch.cat([out_cnv5_3, out_cnv2_2],1)
    
    out_cnv6_1 = self.cnv6_1(in_cnv6_1)
    out_cnv6_2 = self.cnv6_2(out_cnv6_1)
    
    # add skip connection
    in_cnv7_1 = torch.cat([out_cnv6_2, out_cnv1_2],1)
    
    out_cnv7_1 = self.cnv7_1(in_cnv7_1)
    out_cnv7_2 = self.cnv7_2(out_cnv7_1)
    
    out_cnv8_1 = self.cnv8_1(out_cnv7_2)
    
    return out_cnv8_1

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True, img_size=224):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
            
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate

        self.mean_const = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        self.std_const = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

        self.resize = resize
        self.img_size = img_size

    def forward(self, mpi_pred, dep): # input, target
        rgba_layers = mpi_from_net_output(mpi_pred, dep)
        rel_pose = torch.matmul(dep['tgt_img_cfw'], dep['ref_img_wfc'])
        
        input  = mpi_render_view_torch(rgba_layers, rel_pose, dep['mpi_planes'][0], dep['intrinsics'])
        target = dep['tgt_img']

        input = input.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
        
        input = (input-self.mean_const) / self.std_const
        target = (target-self.mean_const) / self.std_const
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(self.img_size, self.img_size), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(self.img_size, self.img_size), align_corners=False)
        x = input
        y = target
        loss = torch.nn.functional.l1_loss(x, y)
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y) / (1+i)
        return loss


if __name__ == '__main__':
    import torch
    model = StereoMagnificationModel(4)
    print(model)
    x = torch.rand(2,15,224,224).cuda()
    model.cuda()
    out = model(x)
    print(out.shape)