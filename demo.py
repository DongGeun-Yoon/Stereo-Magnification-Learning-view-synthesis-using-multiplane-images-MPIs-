import os
import PIL
import base64
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser

from dataset import RealEstateDataset
from networks import StereoMagnificationModel

from utils import *

img_size = 224
batch_size = 1
checkpoint = 'debug/checkpoint.tar' 
data_dir = 'real-estate-10k-run'
save_dir = 'mpi_vision'

parser = ArgumentParser(description="Demo for MPIs")

parser.add_argument('--data_dir', default=data_dir, type=str, help="Directory of real-estate data")
parser.add_argument('--checkpoint', default=checkpoint, type=str, help="Directory of load checkpoint")
parser.add_argument('--img_size', default=img_size, type=int, help="training data image resolution")
parser.add_argument('--batch_size', default=batch_size, type=int, help="Mini batch size")


def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return "data:image/png;base64," + base64.b64encode(img_file.read()).decode('utf-8')

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    # load checkpoint
    ckt = torch.load(checkpoint)
    num_planes = ckt['num_planes']
    model = StereoMagnificationModel(num_mpi_planes=num_planes)
    model.load_state_dict(ckt['state_dict'])

    # Move to GPU, if available
    model = model.to(device)
    model.eval()

    # Custom dataloaders
    valid_dataset = RealEstateDataset(data_dir, img_size=img_size, is_valid=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    for i, (img, dep) in enumerate(valid_loader):
        img = img.type(torch.FloatTensor).to(device) 
        target = dep['tgt_img'].to(device)
        # Forward prop.
        out = model(img)  
        rgba_layers = mpi_from_net_output(out, dep)
        rel_pose = torch.matmul(dep['tgt_img_cfw'], dep['ref_img_wfc']).to(device)
        pred_image = mpi_render_view_torch(rgba_layers, rel_pose, dep['mpi_planes'][0], dep['intrinsics']).to(device)

        # save image
        for j in range(num_planes):
            save_image( rgba_layers[0, :, :, j,:].cpu(), os.path.join(save_dir, "mpi{}.png".format(  ("0" + str(j))[-2:]  )))

        if i == 0:
            break

    image_srcs = [get_base64_encoded_image(os.path.join(save_dir, 'mpi{}.png'.format( ("0" + str(i))[-2:] ))) for i in range(num_planes)]

    # make mpi viewer
    with open("./mpi_vision/deepview-mpi-viewer-template.html", "r") as template_file:
        template_str = template_file.read()

    MPI_SOURCES_DATA = ",".join(['\"' + img_src + '\"' for img_src in image_srcs])
    template_str = template_str.replace("const mpiSources = MPI_SOURCES_DATA;", "const mpiSources = [{}];".format(MPI_SOURCES_DATA))

    with open("./deepview-mpi-viewer.html", "w") as output_file:
        output_file.write(template_str)
