import torch
import torch.nn as nn
import torchvision
from vcop import VCOPModel, training_step_fn
from phoenix_data import PhoenixVCOPDataset, data_from_file
from training import train_model

train_data = data_from_file('data/train.annotations_only')
dev_data = data_from_file('data/dev.annotations_only')

train_dataset = PhoenixVCOPDataset(train_data, videos_root='../../data/phoenix2014t/videos', num_clips=3, clip_length=8, interval=4, return_name=False, resize_factor=0.5, deterministic=False)
dev_dataset = PhoenixVCOPDataset(dev_data, videos_root='../../data/phoenix2014t/videos', num_clips=3, clip_length=8, interval=4, return_name=False, resize_factor=0.5, deterministic=True)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, drop_last=True, shuffle=True, num_workers=3)
validloader = torch.utils.data.DataLoader(dev_dataset, batch_size=8, drop_last=False, shuffle=False, num_workers=3)

cnn3d = torchvision.models.video.r2plus1d_18(pretrained=False)
cnn3d = nn.Sequential(*list(cnn3d.children())[:-2])
model = VCOPModel(feature_extractor=cnn3d, feature_size=512, num_clips=3, dropout_prob=0.1)
model.optimizer = torch.optim.Adam(model.parameters(), 1e-5)
model.to('cuda')

train_model(model, training_step_fn, trainloader, validloader, epochs=1, print_every=1)