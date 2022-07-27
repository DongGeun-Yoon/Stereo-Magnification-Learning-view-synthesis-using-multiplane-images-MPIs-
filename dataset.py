import os
import PIL
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from  utils import *

class RealEstateDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_path, is_valid=False, min_dist=16e3, max_dist=500e3, img_size=(512, 288), num_planes=32):
    self.is_valid = is_valid
    self.dataset_path = dataset_path
    self.min_dist = min_dist
    self.max_dist = max_dist
    self.num_planes = num_planes
    if isinstance(img_size, int):
      self.img_size = (img_size, img_size)
    else:
      self.img_size = img_size

    metadataBasePath = os.path.join(dataset_path, "RealEstate10K", "test" if is_valid else "train")
    self.scenes = [self.get_data(os.path.join(metadataBasePath, subPath)) for subPath in os.listdir(metadataBasePath)]
  
  def get_data(self, dataID):
    txt_id = open(dataID)
    youtube_url = txt_id.readline().strip()
    youtubeIDOffset = youtube_url.find("/watch?v=") + len('/watch?v=')
    youtube_id = youtube_url[youtubeIDOffset:] 
    youtube_data = pickle.load(open(os.path.join(self.dataset_path, 'transcode', youtube_id, 'case.pkl'), 'rb'))

    timestamps = []
    intrinsics = []
    poses = []
    img_path = []
    for data in youtube_data:
      if 'imgPath' in data.keys():
        timestamps.append(data['timeStamp'])
        intrinsics.append(data['intrinsics'])
        poses.append([data['pose'][0:4], data['pose'][4:8], data['pose'][8:12], [0., 0., 0., 1.]])
        img_path.append(data['imgPath'])
   
    return {
          'text_id': dataID.split('/')[-1],
          'youtube_id': youtube_id,
          'timestamps': timestamps,
          'intrinsics': intrinsics,
          'poses': poses, # poses is world to camera (c_f_w o w_t_c)
          'imgpath': img_path}

  def __len__(self):
    return len(self.scenes)
  
  def new_empty(self):
    return []
  
  def _draw(self, scene):
    img_range = range(len(scene['timestamps']))
    ref_img_idx = np.random.choice(img_range)
    base_timestamp = scene['timestamps'][ref_img_idx]
    near_range = list(filter(lambda i: abs(base_timestamp - scene['timestamps'][i]) >= self.min_dist and  abs(base_timestamp - scene['timestamps'][i]) <= self.max_dist, img_range))    
    assert(len(near_range) >= 2)

    src_img_idx = np.random.choice(near_range)
    tgt_img_idx = np.random.choice([i for i in near_range if i != src_img_idx])
    return [ref_img_idx, src_img_idx, tgt_img_idx]
  
  def __getitem__(self, i):
    scene = self.scenes[i]
    if self.is_valid:
      indexes = [0,1,2]
    else:
      indexes = self._draw(scene)
    
    selected_metadata = []
    for index in indexes:
      intrinsics = torch.Tensor(scene['intrinsics'][index]).to(device)
      new_intrinsics = make_intrinsics_matrix( 
          self.img_size[0] * intrinsics[0], # fx
          self.img_size[1] * intrinsics[1], # fy
          self.img_size[0] * intrinsics[2], # cx
          self.img_size[1] * intrinsics[3]  # cy
      )

      image_path = scene['imgpath'][index]
      img = PIL.Image.open(image_path).convert('RGB')
      scaled_image = img.resize(self.img_size)
      tensor_image = preprocess_image_torch(torch.Tensor(np.array(scaled_image)).to(device)/255.0)

      selected_metadata.append({
        'timestamp': scene['timestamps'][index],
        'intrinsics': new_intrinsics,
        'pose': torch.Tensor(scene['poses'][index]).to(device),
        'image': tensor_image
      })
    
    ref_img = selected_metadata[0] # this one will be fed to the nn without processing
    src_img = selected_metadata[1] # this will be fed to the nn as psv
    tgt_img = selected_metadata[2] # this is the dependent variable what the output should be
    
    psv_planes = torch.Tensor(inv_depths(1, 100, self.num_planes)).to(device)
    curr_pose = torch.matmul(src_img['pose'], torch.inverse(ref_img['pose']))
    curr_psv = plane_sweep_torch_one(src_img['image'], psv_planes, curr_pose, src_img['intrinsics'])
    
    net_input = torch.cat([torch.unsqueeze(ref_img['image'], 0), curr_psv], 3)
    dep_var = {
        'tgt_img_cfw': tgt_img['pose'],
        'tgt_img': tgt_img['image'],
        'ref_img': ref_img['image'],
        'ref_img_wfc': torch.inverse(ref_img['pose']),
        'intrinsics': src_img['intrinsics'],
        'mpi_planes': psv_planes,
        'data_id': scene['text_id']
    }
    return [torch.squeeze(net_input).permute([2, 0, 1]), dep_var]


if __name__ == '__main__':
  train_data = RealEstateDataset("./real-estate-10k-run")
  print(len(train_data))
  