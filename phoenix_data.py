import torch
import torchvision

class PhoenixDataset(torch.utils.data.Dataset):
    def __init__(self, data, videos_root=None, source_mode='gloss', target_mode='text', self_sup_video_length=32, video_scale_factor=1, return_name=False):
        super(PhoenixDataset, self).__init__()

        # data is assumed to be a list of dicts {'name': 'train/...', 'text': string, 'gloss': string, 'sign': tensor, 'video': tensor}
        # all keys are not required, depending on source_mode and target_mode
        # 'video' must be an uint8 tensor, of shape (length, height, width, channels)
        # 'sign' corresponds to embeddings (in the SLTT paper they are named like this...)
        
        self.data = data
        self.videos_root = videos_root
        self.source_mode = source_mode # can be 'gloss', 'text', 'video' or 'sign'
        self.target_mode = target_mode # can be 'gloss', 'text', 'video', 'sign' or 'self-sup'
        self.self_sup_video_length = self_sup_video_length
        self.video_scale_factor = video_scale_factor
        self.return_name = return_name # If True, returns the 'name' attribute of the annotation
        
        self.video_mean = torch.Tensor([0.43216, 0.394666, 0.37645])[:, None, None, None]
        self.video_std = torch.Tensor([0.22803, 0.22145, 0.216989])[:, None, None, None] # https://pytorch.org/docs/stable/torchvision/models.html#video-classification
        
        assert (self.target_mode == 'self-sup' and self.source_mode == 'video')\
                or self.target_mode != 'self-sup', 'Self supervised target mode is only supported for "video" as source mode'
    
    def preprocess_video(self, video):
        video = (video.float()/255-self.video_mean) / self.video_std
        if self.video_scale_factor != 1:
            video = F.interpolate(video, scale_factor=self.video_scale_factor, mode='bilinear')
        return video
    
    def get_self_sup_sample(self, video):

        # video is assumed to have shape (channels, length, height, width)

        num_frames = video.shape[1]
            
        # clip video
        t0 = torch.randint(low=0, high=max(1, num_frames-self.self_sup_video_length), size=(1,))
        t1 = t0 + self.self_sup_video_length
        t1 = min(t1, num_frames)    
        video = video[:, t0:t1, :, :]
        
        video = self.preprocess_video(video)
            
        flip = torch.rand(1) < 0.5
        if flip:
            video = torch.flip(video, dims=(1,))
            
        # pad video
        num_frames = video.shape[1]
        video = F.pad(video, (0, 0, 0, 0, 0, self.self_sup_video_length-num_frames, 0, 0))
                        
        return video, int(flip)

    def get_video(self, idx):

        if 'video' in self.data[idx].keys():
            video = self.data[idx]['video']

        else:
            video_path = os.path.join(self.videos_root, self.data[idx]['name'] + '.mp4')
            video, audio, info = torchvision.io.read_video(video_path)

        if video.shape[-1] == 3:
            # shape is probabaly now (length, height, width, channels)
            video = video.permute(3, 0, 1, 2) # shape is now (channels, length, height, width)

        return video
    
    def __getitem__(self, idx):
                
        if self.source_mode == 'video':
            src = self.get_video(idx)
            if not self.target_mode == 'self-sup': # if self.target_mode == 'self-sup', the preprocessing is alreay done in get_self_sup_sample
                src = self.preprocess_video(src)
        else:
            src = self.data[idx][self.source_mode]
        
        if self.target_mode == 'self-sup':
            src, tgt = self.get_self_sup_sample(src)
        elif self.target_mode == 'video':
            tgt = self.get_video(idx)
            tgt = self.preprocess_video(tgt)
        else:
            tgt = self.data[idx][self.target_mode]
        
        if self.return_name:
            return src, tgt, self.data[idx]['name']
        return src, tgt
    
    def __len__(self):
        return len(self.data)