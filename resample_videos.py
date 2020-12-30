import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
import gzip
import pickle
from tqdm import tqdm

video_root = '../../data/phoenix2014t/videos/'
subset = 'test'

with gzip.open(f'../../data/phoenix2014t/{subset}.annotations_only', 'rb') as f:
    annotations = pickle.load(f)

new_annotations = []
for i, annotation in enumerate(tqdm(annotations)):
    video, audio, info = torchvision.io.read_video(os.path.join(video_root, annotation['name']+'.mp4'))
    video = video.permute(3, 0, 1, 2).float()
    video = F.interpolate(video, scale_factor=0.5, mode='bilinear')
    video = video.clamp(min=0, max=255)
    video = video.type(torch.uint8)
    new_annotations.append({**annotation, 'video': video})

with open(f'../../data/phoenix2014t/{subset}_130x105.pkl', 'wb') as output:
    pickle.dump(new_annotations, output, 3)