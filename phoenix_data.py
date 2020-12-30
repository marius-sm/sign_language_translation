import torch
import torch.nn.functional as F
import torchvision
import os

class PhoenixDataset(torch.utils.data.Dataset):
    def __init__(self,
        data,
        videos_root=None,
        source_mode='gloss',
        target_mode='text',
        rescale_mean=[0.43216, 0.394666, 0.37645], # from https://pytorch.org/docs/stable/torchvision/models.html#video-classification
        rescale_std=[0.22803, 0.22145, 0.216989],  # from https://pytorch.org/docs/stable/torchvision/models.html#video-classification
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
        self.rescale_mean = torch.Tensor(rescale_mean)[:, None, None, None]
        self.rescale_std = torch.Tensor(rescale_std)[:, None, None, None]
        self.resize_factor = resize_factor
        self.return_name = return_name # If True, returns the 'name' attribute of the annotation

        assert self.source_mode in ['gloss', 'text', 'video', 'sign', 'embedding'], 'Invalid source mode'
        assert self.target_mode in ['gloss', 'text', 'video', 'sign', 'embedding', None], 'Invalid target mode'

    def rescale_video(self, video):
        return (video.float()/255-self.rescale_mean) / self.rescale_std
    
    def resize_video(self, video):
        if self.resize_factor != 1:
            video = F.interpolate(video, scale_factor=self.resize_factor, mode='bilinear')
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
            video = video.permute(3, 0, 1, 2) # shape is now (channels, length, height, width)

        return video
    
    def __getitem__(self, idx):
                
        if self.source_mode == 'video':
            src = self.get_video(idx)
            src = self.rescale_video(src)
            src = self.resize_video(src)
        else:
            src = self.data[idx][self.source_mode]
        
        if self.target_mode == 'video':
            tgt = self.get_video(idx)
            tgt = self.rescale_video(tgt)
            tgt = self.resize_video(tgt)
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
    def __init__(self, data, videos_root=None, clip_length=32, **kwargs):
        super(PhoenixTimeArrowDataset, self).__init__(data, videos_root, source_mode='video', target_mode=None, **kwargs)
        self.clip_length = clip_length

    def __getitem__(self, idx):
        video = super(PhoenixTimeArrowDataset, self).__getitem__(idx) # video should have shape (channels, length, height, width)
        num_frames = video.shape[1]
            
        # clip video
        t0 = torch.randint(low=0, high=max(1, num_frames-self.clip_length), size=(1,))
        t1 = t0 + self.clip_length
        t1 = min(t1, num_frames)    
        video = video[:, t0:t1, :, :]
                    
        flip = torch.rand(1) < 0.5
        if flip:
            video = torch.flip(video, dims=(1,))
            
        # pad video
        num_frames = video.shape[1]
        video = F.pad(video, (0, 0, 0, 0, 0, self.clip_length-num_frames, 0, 0))
                        
        return video, int(flip)


class PhoenixVCOPDataset(PhoenixDataset):
    def __init__(self, data, videos_root=None, video_scale_factor=1, return_name=False):
        super(PhoenixVCOPDataset, self).__init__(data, videos_root, video_scale_factor=video_scale_factor, return_name=return_name, source_mode='video', target_mode=None)




