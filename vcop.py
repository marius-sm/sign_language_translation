import torch
import torch.nn as nn
import torch.nn.functional as F
from phoenix_data import data_from_file, PhoenixVCOPDataset
import math

class VCOPModel(nn.Module):
    def __init__(self, feature_extractor, num_clips, feature_size=512, dropout_prob=0.5):
        super(VCOPModel, self).__init__()

        self.num_clips = num_clips
        self.num_orders = math.factorial(num_clips)

        self.feature_size = feature_size
        self.feature_extractor = feature_extractor
        self.pooling = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.flatten = nn.Flatten(start_dim=1)

        self.fc1 = nn.Linear(self.feature_size*2, 512)
        num_pairs = (num_clips*(num_clips-1))//2
        self.fc2 = nn.Linear(512*num_pairs, self.num_orders)

        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, clip_list):

        features = []
        for clip_batch in clip_list:
            features.append(self.flatten(self.pooling(self.feature_extractor(clip_batch))))

        pairs = []  # pairwise concatenation of features
        for i in range(self.num_clips):
            for j in range(i+1, self.num_clips):
                pairs.append(torch.cat([features[i], features[j]], dim=1))

        x = [self.fc1(p) for p in pairs]
        x = [self.relu(y) for y in x]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        x = self.fc2(x)  # logits

        return x