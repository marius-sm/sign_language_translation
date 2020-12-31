import torch
import torch.nn.functional as F
import torchvision
import os
import itertools
import time
import gzip
import pickle

def data_from_file(filename):
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

class PhoenixDataset(torch.utils.data.Dataset):
    def __init__(self,
        data,
        videos_root=None,
        source_mode='gloss',
        target_mode='text',
        normalize=True,
        normalize_mean=[0.43216, 0.394666, 0.37645], # from https://pytorch.org/docs/stable/torchvision/models.html#video-classification
        normalize_std=[0.22803, 0.22145, 0.216989],  # from https://pytorch.org/docs/stable/torchvision/models.html#video-classification
        resize_factor=1,
        return_name=False
        ):
        super(PhoenixDataset, self).__init__()

        # data is assumed to be a list of dicts {'name': 'train/...', 'text': string, 'gloss': string, 'sign': tensor, 'video': tensor}
        # all keys are not required, depending on source_mode and target_mode
        # if data contains a 'video' field, it must be an uint8 tensor, of shape (channels, length, height, width)
        # 'sign' corresponds to embeddings (in the SLTT paper they are named like this...)
        
        self.data = data
        self.videos_root = videos_root
        self.source_mode = source_mode # can be 'gloss', 'text', 'video' or 'sign'
        self.target_mode = target_mode # can be 'gloss', 'text', 'video', 'sign' or None

        self.normalize = normalize
        self.normalize_mean = torch.Tensor(normalize_mean)[:, None, None, None]
        self.normalize_std = torch.Tensor(normalize_std)[:, None, None, None]
        self.resize_factor = resize_factor
        self.return_name = return_name # If True, returns the 'name' attribute of the annotation

        assert self.source_mode in ['gloss', 'text', 'video', 'sign', 'embedding'], 'Invalid source mode'
        assert self.target_mode in ['gloss', 'text', 'video', 'sign', 'embedding', None], 'Invalid target mode'

    def rescale_video(self, video, mean, std):
        # rescaling is slow
        return video.float()/(255*std) - (mean / std)
    
    def resize_video(self, video, factor):
        if factor == 1: return video
        video = video.float()
        video = F.interpolate(video, scale_factor=factor, mode='bilinear')
        return video

    def get_video(self, idx):

        if 'video' in self.data[idx].keys():
            video = self.data[idx]['video']

        else:
            assert self.videos_root is not None, 'video not found in data, please provide a valid "videos_root"'
            video_path = os.path.join(self.videos_root, self.data[idx]['name'] + '.mp4')
            video, audio, info = torchvision.io.read_video(video_path)

        if video.shape[-1] == 3:
            # shape is probabaly (length, height, width, channels)
            video = video.permute(3, 0, 1, 2) # shape is now (channels, length, height, width). This is fast

        return video
    
    def __getitem__(
        self, idx,
        normalize=None,
        normalize_mean=None,
        normalize_std=None,
        resize_factor=None
        ):

        normalize = self.normalize if normalize is None else normalize
        normalize_mean = self.normalize_mean if normalize_mean is None else normalize_mean
        normalize_std = self.normalize_std if normalize_std is None else normalize_std
        resize_factor = self.resize_factor if resize_factor is None else resize_factor
                
        if self.source_mode == 'video':
            src = self.get_video(idx)
            if normalize:
                src = self.rescale_video(src, normalize_mean, normalize_std)
            src = self.resize_video(src, resize_factor)
        else:
            src = self.data[idx][self.source_mode]
        
        if self.target_mode == 'video':
            tgt = self.get_video(idx)
            if normalize:
                tgt = self.rescale_video(tgt, normalize_mean, normalize_std)
            tgt = self.resize_video(tgt, resize_factor)
        elif self.target_mode is not None:
            tgt = self.data[idx][self.target_mode]
        else:
            tgt = None
        
        if self.return_name:
            if self.target_mode is not None:
                return src, tgt, self.data[idx]['name']
            else:
                return src, self.data[idx]['name']
        if self.target_mode is not None:
            return src, tgt
        return src
    
    def __len__(self):
        return len(self.data)

class PhoenixTimeArrowDataset(PhoenixDataset):
    def __init__(self, data, clip_length=32, **kwargs):

        super(PhoenixTimeArrowDataset, self).__init__(data, source_mode='video', target_mode=None, **kwargs)

        self.clip_length = clip_length

    def __getitem__(self, idx):
        video = super(PhoenixTimeArrowDataset, self).__getitem__(idx, normalize=False, resize_factor=1) # video should have shape (channels, length, height, width)
        num_frames = video.shape[1]
            
        # clip video
        t0 = torch.randint(low=0, high=max(1, num_frames-self.clip_length), size=(1,)).item()
        t1 = t0 + self.clip_length
        t1 = min(t1, num_frames)    
        video = video[:, t0:t1, :, :]
                    
        flip = torch.rand(1) < 0.5
        if flip:
            video = torch.flip(video, dims=(1,))

        video = self.resize_video(self.rescale_video(video, self.normalize_mean, self.normalize_std), self.resize_factor)

        # pad video
        num_frames = video.shape[1]
        video = F.pad(video, (0, 0, 0, 0, 0, self.clip_length-num_frames, 0, 0))
                        
        return video, int(flip)


class PhoenixVCOPDataset(PhoenixDataset):
    def __init__(self, data, num_clips, clip_length, interval, deterministic=False, **kwargs):

        super(PhoenixVCOPDataset, self).__init__(data, source_mode='video', target_mode=None, **kwargs)

        self.min_video_length = num_clips * clip_length + (num_clips-1)*interval
        self.num_clips = num_clips
        self.clip_length = clip_length
        self.interval = interval
        self.deterministic = deterministic

        self.orders = list(itertools.permutations(list(range(self.num_clips))))

    def __getitem__(self, idx):
        i = idx
        while True:
            video = super(PhoenixVCOPDataset, self).__getitem__(i, normalize=False, resize_factor=1) # video should have shape (channels, length, height, width)
            num_frames = video.shape[1]
            if num_frames >= self.min_video_length:
                break
            i = torch.randint(low=0, high=self.__len__(), size=(1,)).item()
            print(f'Sample {idx} does not have enough frames (min {self.min_video_length})')

        clips = []
        t0 = torch.randint(low=0, high=max(1, num_frames-self.min_video_length), size=(1,)).item()
        if self.deterministic:
            t0 = idx%max(1, num_frames-self.min_video_length)
        for j in range(self.num_clips):
            clips.append(video[:, t0:t0+self.clip_length])
            t0 += self.clip_length + self.interval

        order_index =  torch.randint(low=0, high=len(self.orders), size=(1,)).item()
        if self.deterministic:
            order_index = idx%len(self.orders)
        order = self.orders[order_index]
        clips = [self.resize_video(self.rescale_video(clips[j], self.normalize_mean, self.normalize_std), self.resize_factor) for j in order]

        return clips, order_index


