import torch
import torch.nn.functional as F
import torchvision
import os
import itertools
import time
import gzip
import pickle
import numpy as np
import imgaug.augmenters as iaa
import imgaug

def data_from_file(filename):
    is_gzip = False
    with open(filename, 'rb') as f:
        is_gzip = f.read(2) == b'\x1f\x8b'
    if is_gzip:
        with gzip.open(filename, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    return data



def augment_video(video):
    # video has shape (3, L, h, w)
    
    if isinstance(video, torch.Tensor):
        video = video.permute(1, 2, 3, 0).numpy()
    imgaug.seed(torch.randint(0, 99999999, size=(1,)).item())
    aug = iaa.Sequential([  
        iaa.Crop(percent=(0, 0.1)),
        iaa.LinearContrast((0.75, 1.2)),
        iaa.Multiply((0.75, 1.333), per_channel=0.5),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-10, 10),
            shear=(-10, 10)
        )
    ])
    
    aug_det = aug.to_deterministic()
    results = np.zeros(video.shape,  video.dtype)
    for i in range(video.shape[0]):
        results[i] = aug_det.augment_image(video[i])
        
    return torch.from_numpy(results).permute(3, 0, 1, 2)

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
        return_name=False,
        video_filter=lambda v: True,
        store_videos=False,
        transform=None
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
        self.video_filter = video_filter
        self.store_videos = store_videos
        self.storage = {}
        self.transform = transform

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

        if self.store_videos and idx in self.storage.keys():
            video = self.storage[idx]
            if self.transform is not None:
                return self.transform(video)
            return video
        
        if 'video' in self.data[idx].keys():
            video = self.data[idx]['video']

        else:
            assert self.videos_root is not None, 'video not found in data, please provide a valid "videos_root"'
            video_path_mp4 = os.path.join(self.videos_root, self.data[idx]['name'] + '.mp4')
            if os.path.isfile(video_path_mp4):
                video, audio, info = torchvision.io.read_video(video_path_mp4)
            else:
                video_path_npy = os.path.join(self.videos_root, self.data[idx]['name'] + '.npy')
                if os.path.isfile(video_path_npy):
                    video = torch.from_numpy(np.load(video_path_npy))
                else:
                    raise FileNotFoundError(f'File {video_path_npy[:-4]}.mp4|npy not found')

        if video.shape[-1] == 3:
            # shape is probabaly (length, height, width, channels)
            video = video.permute(3, 0, 1, 2) # shape is now (channels, length, height, width). This is fast

        if self.store_videos:
            self.storage[idx] = video.clone()
            
        if self.transform is not None:
            return self.transform(video)
        
        return video
    
    def get_video_with_filter(self, idx):
        
        i = idx
        while True:
            video = self.get_video(i) # video should have shape (channels, length, height, width)
            if self.video_filter(video):
                break
            i = torch.randint(low=0, high=self.__len__(), size=(1,)).item()
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
            src = self.get_video_with_filter(idx)
            if normalize:
                src = self.rescale_video(src, normalize_mean, normalize_std)
            src = self.resize_video(src, resize_factor)
        else:
            src = self.data[idx][self.source_mode]
        
        if self.target_mode == 'video':
            tgt = self.get_video_with_filter(idx)
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
    def __init__(self, data, clip_length=32, deterministic=False, **kwargs):

        super(PhoenixTimeArrowDataset, self).__init__(data, source_mode='video', target_mode=None, **kwargs)

        self.clip_length = clip_length
        self.deterministic = deterministic

    def __getitem__(self, idx):
        video = super(PhoenixTimeArrowDataset, self).__getitem__(idx, normalize=False, resize_factor=1) # video should have shape (channels, length, height, width)
        num_frames = video.shape[1]
            
        # clip video
        if not self.deterministic:
            t0 = torch.randint(low=0, high=max(1, num_frames-self.clip_length), size=(1,)).item()
        else:
            t0 = idx%max(1, num_frames-self.clip_length)
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
    def __init__(self, data, num_clips, clip_length, interval, deterministic=False, verbose=False, **kwargs):

        super(PhoenixVCOPDataset, self).__init__(data, source_mode='video', target_mode=None, **kwargs)

        self.min_video_length = num_clips * clip_length + (num_clips-1)*interval
        self.num_clips = num_clips
        self.clip_length = clip_length
        self.interval = interval
        self.deterministic = deterministic
        self.verbose = verbose

        self.orders = list(itertools.permutations(list(range(self.num_clips))))

    def __getitem__(self, idx):
        i = idx
        while True:
            video = super(PhoenixVCOPDataset, self).__getitem__(i, normalize=False, resize_factor=1) # video should have shape (channels, length, height, width)
            num_frames = video.shape[1]
            if num_frames >= self.min_video_length:
                break
            i = torch.randint(low=0, high=self.__len__(), size=(1,)).item()
            if self.verbose:
                print(f'Sample {idx} does not have enough frames (min {self.min_video_length})')

        clips = []
        if not self.deterministic:
            t0 = torch.randint(low=0, high=max(1, num_frames-self.min_video_length), size=(1,)).item()
        else:
            t0 = idx%max(1, num_frames-self.min_video_length)
        for j in range(self.num_clips):
            clips.append(video[:, t0:t0+self.clip_length])
            t0 += self.clip_length + self.interval

        if not self.deterministic:
            order_index =  torch.randint(low=0, high=len(self.orders), size=(1,)).item()
        else:
            order_index = idx%len(self.orders)
        order = self.orders[order_index]
        clips = [self.resize_video(self.rescale_video(clips[j], self.normalize_mean, self.normalize_std), self.resize_factor) for j in order]

        return clips, order_index


class PhoenixDistancePredictionDataset(PhoenixDataset):
    def __init__(self, data, clip_length, short_interval, long_interval, deterministic=False, verbose=False, **kwargs):

        super(PhoenixDistancePredictionDataset, self).__init__(data, source_mode='video', target_mode=None, **kwargs)

        self.min_video_length = 3 * clip_length + short_interval + long_interval
        self.clip_length = clip_length
        self.short_interval = short_interval
        self.long_interval = long_interval
        self.deterministic = deterministic
        self.verbose = verbose

    def __getitem__(self, idx):
        i = idx
        while True:
            video = super(PhoenixDistancePredictionDataset, self).__getitem__(i, normalize=False, resize_factor=1) # video should have shape (channels, length, height, width)
            num_frames = video.shape[1]
            if num_frames >= self.min_video_length:
                break
            i = torch.randint(low=0, high=self.__len__(), size=(1,)).item()
            if self.verbose:
                print(f'Sample {idx} does not have enough frames (min {self.min_video_length})')

        clips = []

        if not self.deterministic:
            t0 = torch.randint(low=0, high=max(1, num_frames-self.min_video_length), size=(1,)).item()
        else:
            t0 = idx%max(1, num_frames-self.min_video_length)

        if not self.deterministic:
            short_interval_first = bool(torch.randint(low=0, high=2, size=(1,)).item())
        else:
            short_interval_first = bool(idx%2)

        if short_interval_first:
            close = video[:, t0:t0+self.clip_length]
            t0 += self.clip_length + self.short_interval
            anchor = video[:, t0:t0+self.clip_length]
            t0 += self.clip_length + self.long_interval
            far = video[:, t0:t0+self.clip_length]
        else:
            far = video[:, t0:t0+self.clip_length]
            t0 += self.clip_length + self.long_interval
            anchor = video[:, t0:t0+self.clip_length]
            t0 += self.clip_length + self.short_interval
            close = video[:, t0:t0+self.clip_length]

        anchor = self.resize_video(self.rescale_video(anchor, self.normalize_mean, self.normalize_std), self.resize_factor)
        close = self.resize_video(self.rescale_video(close, self.normalize_mean, self.normalize_std), self.resize_factor)
        far = self.resize_video(self.rescale_video(far, self.normalize_mean, self.normalize_std), self.resize_factor)

        return anchor, close, far
