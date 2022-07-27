import os
import PIL
import base64
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
from networks import StereoMagnificationModel
from utils import *

Image1 = 'mpi_vision/bananas1.jpg'
Image2 = 'mpi_vision/bananas2.jpg'
Pose1 = '0.999947548 -0.001357046 -0.010151989 0.359417914 0.001413203 0.999983728 0.005526535 -0.003794850 0.010144324 -0.005540592 0.999933183 0.124850837'
Pose2 = '0.999962628 -0.002106981 -0.008386925 0.376296459 0.002151508 0.999983609 0.005303705 -0.001512435 0.008375612 -0.005321552 0.999950767 0.124671650'
xoffset = 0.017
yoffset = 0.0
zoffset = 0.0

fx = 0.5
fy = 0.5
render_multiples = '-2,-1.5,-1,-0.5,0,0.5,1,1.5,2'
render = True

xshift = 0
yshift = 0

img_size = (512, 288)
batch_size = 1
checkpoint = 'pretrained/tf.pth'
save_dir = 'mpi_vision'

parser = ArgumentParser(description="Demo for MPIs")

parser.add_argument('--image1', default=Image1, type=str, help="Directory of ref image")
parser.add_argument('--image2', default=Image2, type=str, help="Directory of src image")
parser.add_argument('--pose1', default=Pose1, type=str, help="ref image pose")
parser.add_argument('--pose2', default=Pose2, type=str, help="src image pose")
parser.add_argument('--xoffset', default=xoffset, type=float, help="xoffset")
parser.add_argument('--yoffset', default=yoffset, type=float, help="yoffset")
parser.add_argument('--zoffset', default=zoffset, type=float, help="zoffset")
parser.add_argument('--render_multiples', default=render_multiples, type=str, help="rendering parameters")
parser.add_argument('--render', default=render, type=bool, help="render")
parser.add_argument('--fx', default=fx, type=float, help="Focal length")
parser.add_argument('--fy', default=fy, type=float, help="Focal length")
parser.add_argument('--xshift', default=xshift, type=float, help="Horizontal pixel shift")
parser.add_argument('--yshift', default=yshift, type=float, help="Vertical pixel shift")

parser.add_argument('--save_dir', default=save_dir, type=str, help="Directory of save rendered view")
parser.add_argument('--checkpoint', default=checkpoint, type=str, help="Directory of load checkpoint")
parser.add_argument('--batch_size', default=batch_size, type=int, help="Mini batch size")


def shift_image(image, x, y):
    height, width, _ = image.shape # [h, w, 3]
    x = int(round(x))
    y = int(round(y))
    dtype = image.dtype
    if x > 0:
        image = np.concatenate((np.zeros([height, x, 3], dtype=dtype), image[:, :(width-x)]), axis=1)
    elif x < 0:
        image = np.concatenate((image[:, -x:], np.zeros([height, -x, 3], dtype=dtype)), axis=1)
    if y > 0:
        image = np.concatenate((np.zeros([y, width, 3], dtype=dtype), image[:(height-y), :]), axis=0)
    elif y < 0:
        image = np.concatenate((image[-y:, :], np.zeros([-y, width, 3], dtype=dtype)), axis=0)
    return image

def load_image(path, padx, pady, xshift, yshift):
    image = PIL.Image.open(path).convert('RGB')
    image = np.array(image)
    padded = np.pad(image, ((padx, padx), (pady, pady), (0,0)))
    image = shift_image(padded, xshift, yshift)
    return image

def crop_to_multiple(image, size):
    height, width, _ = image.shape
    new_width = width - (width % size)
    new_height = height - (height % size)

    left = (width % size) // 2
    right = new_width + left
    top = (height % size) // 2
    bottom = new_height + top
    return image[top:bottom, left:right]

def pose_from_args(pose):
    values = [float(x) for x in pose.replace(',', ' ').split()]
    assert len(values) == 12
    return torch.Tensor([values[0:4], values[4:8], values[8:12], [0.0, 0.0, 0.0, 1.0]])

def get_inputs(padx, pady):
    inputs = {}
    image1 = load_image(args.image1, padx, pady, 0, 0)
    image2 = load_image(args.image2, padx, pady, -args.xshift, -args.yshift)

    shape1_before_crop = image1.shape # [h, w, 3]
    shape2_before_crop = image2.shape
    image1 = crop_to_multiple(image1, 16)
    image2 = crop_to_multiple(image2, 16)
    shape1_after_crop = image1.shape
    shape2_after_crop = image2.shape
    
    pose_one = pose_from_args(args.pose1).to(device)
    pose_two = pose_from_args(args.pose2).to(device)

    original_width = shape1_before_crop[1] - 2 * padx
    original_height = shape1_before_crop[0] - 2 * pady
    eventual_width = shape1_after_crop[1]
    eventual_height = shape1_after_crop[0]
    
    fx = original_width * float(args.fx)
    fy = original_height * float(args.fy)

    cx = eventual_width * 0.5
    cy = eventual_height * 0.5
    
    intrinsics = torch.Tensor([[fx, 0.0, cx],
                               [0.0, fy, cy],
                               [0.0, 0.0, 1.0]]).to(device)
    tensor_image1 = preprocess_image_torch(torch.Tensor(image1)/255.).to(device)
    tensor_image2 = preprocess_image_torch(torch.Tensor(image2)/255.).to(device)

    psv_planes = torch.Tensor(inv_depths(1, 100, 32)).to(device)
    curr_pose = torch.matmul(pose_two, torch.inverse(pose_one))
    curr_psv = plane_sweep_torch_one(tensor_image2, psv_planes, curr_pose, intrinsics)
    inputs = torch.cat([torch.unsqueeze(tensor_image1, 0), curr_psv], 3)
    dev_var = {
        'tgt_img_cfw': 1,
        'tgt_img': 1,
        'ref_img': tensor_image2,
        'ref_img_wfc': pose_two,
        'intrinsics': intrinsics,
        'mpi_planes': psv_planes
    }
    return inputs.permute([0,3,1,2]), original_width, original_height, dev_var

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return "data:image/png;base64," + base64.b64encode(img_file.read()).decode('utf-8')

if __name__ == '__main__':
    import time
    args = parser.parse_args()
    # load checkpoint
    ckt = torch.load(args.checkpoint)
    num_planes = 32
    model = StereoMagnificationModel(num_mpi_planes=num_planes)
    model.load_state_dict(ckt['state_dict'])

    # Move to GPU, if available
    model = model.to(device)
    model.eval()

    # Custom dataloaders
    max_multiple = 0
    if args.render:
        render_list = [float(x) for x in args.render_multiples.split(',')]
        max_multiple = max(abs(float(m)) for m in render_list)
    padx = int(max_multiple * abs(args.xshift) + 8)
    pady = int(max_multiple * abs(args.yshift) + 8)
    
    print('Padding inputs: padx={}, pady={}, (multiple={})'.format(padx, pady, max_multiple))
    time1 = time.time()
    inputs, original_width, original_height, dep = get_inputs(padx, pady)
    inputs = inputs.type(torch.FloatTensor).to(device) 

    # Forward prop.
    out = model(inputs)  
    rgba_layers = mpi_from_net_output(out, dep)
    
    print(time.time()-time1)
    # save image
    for j in range(num_planes):
        save_image( rgba_layers[0, :, :, j,:].cpu(), os.path.join(args.save_dir, "mpi{}.png".format(  ("0" + str(j))[-2:]  )))

    print(time.time()-time1)

    render = {}
    if args.render:
        print('Rendering new views...')
        for index, multiple in enumerate(render_list):
            m = float(multiple)
            print('    offset:{}'.format(multiple))
            pose = torch.Tensor([[1.0, 0.0, 0.0, -m * args.xoffset], 
                                 [0.0, 1.0, 0.0, -m * args.yoffset],
                                 [0.0, 0.0, 1.0, -m * args.zoffset],
                                 [0.0, 0.0, 0.0, 1.0]]).unsqueeze(0).cuda()
            image = mpi_render_view_torch(rgba_layers, pose, dep['mpi_planes'], dep['intrinsics'].unsqueeze(0))
            
            normalized_img = (image[0,:,:,:3]+1.0)/2.0 # RGB
            normalized_img = normalized_img.permute(2,0,1)
            RGBImg = torchvision.transforms.ToPILImage()( normalized_img )
            RGBImg.save(os.path.join(args.save_dir, 'render_{:02}_{}.png'.format(index, m)))

    image_srcs = [get_base64_encoded_image(os.path.join(args.save_dir, 'mpi{}.png'.format( ("0" + str(i))[-2:] ))) for i in range(num_planes)]

    # make mpi viewer
    with open("./mpi_vision/deepview-mpi-viewer-template.html", "r") as template_file:
        template_str = template_file.read()

    MPI_SOURCES_DATA = ",".join(['\"' + img_src + '\"' for img_src in image_srcs])
    template_str = template_str.replace("const mpiSources = MPI_SOURCES_DATA;", "const mpiSources = [{}];".format(MPI_SOURCES_DATA))

    with open("./deepview-mpi-viewer.html", "w") as output_file:
        output_file.write(template_str)