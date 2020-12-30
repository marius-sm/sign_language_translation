import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
import gzip
import pickle
from tqdm import tqdm

dataset_root = 'phoenix-2014t-videos/'
train_video_paths = sorted(os.listdir(os.path.join(dataset_root, 'train')))
train_video_paths = [os.path.join(os.path.join(dataset_root, 'train'), p) for p in train_video_paths]
test_video_paths = sorted(os.listdir(os.path.join(dataset_root, 'test')))
test_video_paths = [os.path.join(os.path.join(dataset_root, 'test'), p) for p in test_video_paths]
dev_video_paths = sorted(os.listdir(os.path.join(dataset_root, 'dev')))
dev_video_paths = [os.path.join(os.path.join(dataset_root, 'dev'), p) for p in dev_video_paths]

with gzip.open('data/train.annotations_only', 'rb') as f:
    train_annotations = pickle.load(f)
with gzip.open('data/test.annotations_only', 'rb') as f:
    test_annotations = pickle.load(f)
with gzip.open('data/dev.annotations_only', 'rb') as f:
    dev_annotations = pickle.load(f)

new_annotations = []
for annotation in tqdm(dev_annotations[:10]):
    video, audio, info = torchvision.io.read_video(os.path.join('phoenix-2014t-videos/', annotation['name']+'.mp4'))
    video = video.permute(3, 0, 1, 2).float()
    video = F.interpolate(video, scale_factor=0.5, mode='bilinear')
    video = video.clamp(min=0, max=255)
    video = video.type(torch.uint8)
    new_annotations.append({**annotation, 'video': video})

with open(f'data/dev_130x105.pkl', 'wb') as output:
    pickle.dump(new_annotations, output, 3)