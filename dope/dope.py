import os
import sys
import argparse
import os.path as osp
from PIL import Image
import cv2
import numpy as np

import torch
from torchvision.transforms import ToTensor

from model import dope_resnet50, num_joints
import postprocess

import visu

def dope_video(model, video_path, postprocessing='ppi', output_dir='outputs/', save_2d=True, save_3d=True):
    if postprocessing=='ppi':
        sys.path.append('/lcrnet-v2-improved-ppi/')
        try:
            from lcr_net_ppi_improved import LCRNet_PPI_improved
        except ModuleNotFoundError:
            raise Exception('To use the pose proposals integration (ppi) as postprocessing, please follow the readme instruction by cloning our modified version of LCRNet_v2.0 here. Alternatively, you can use --postprocess nms without any installation, with a slight decrease of performance.')

    # load video frames
    video_name = video_path.split("/")[-1].split(".")[0]
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        imlist = [ToTensor()(image).to(device)]
        if ckpt['half']:
            imlist = [im.half() for im in imlist]
        resolution = imlist[0].size()[-2:]

        # forward pass of the dope network
        with torch.no_grad():
            results, avg_features, max_features, max_indices = model(imlist, None)
            results = results[0]
      
        # save embeddings
        emb = torch.cat([avg_features, max_features, max_indices])
        emb_subdir = os.path.join(output_dir, 'embeddings', video_name)
        if not os.path.exists(emb_subdir):
            os.makedirs(emb_subdir)
        emb_path = os.path.join(emb_subdir, f'{count}.pt')
        torch.save(emb, emb_path)

        # postprocess results (pose proposals integration, wrists/head assignment)
        assert postprocessing in ['nms','ppi']
        parts = ['body','hand','face']
        if postprocessing=='ppi':
            res = {k: v.float().data.cpu().numpy() for k,v in results.items()}
            detections = {}
            for part in parts:
                detections[part] = LCRNet_PPI_improved(res[part+'_scores'], res['boxes'], res[part+'_pose2d'], res[part+'_pose3d'], resolution, **ckpt[part+'_ppi_kwargs'])
        else: # nms
            detections = {}
            for part in parts:
                dets, indices, bestcls = postprocess.DOPE_NMS(results[part+'_scores'], results['boxes'], results[part+'_pose2d'], results[part+'_pose3d'], min_score=0.3)
            dets = {k: v.float().data.cpu().numpy() for k,v in dets.items()}
            detections[part] = [{'score': dets['score'][i], 'pose2d': dets['pose2d'][i,...], 'pose3d': dets['pose3d'][i,...]} for i in range(dets['score'].size)]
            if part=='hand':
                for i in range(len(detections[part])):
                    detections[part][i]['hand_isright'] = bestcls<ckpt['hand_ppi_kwargs']['K']

        # assignment of hands and head to body
        detections, body_with_wrists, body_with_head = postprocess.assign_hands_and_head_to_body(detections)
      
        # save 2D results
        if save_2d:
            det_poses2d = {part: np.stack([d['pose2d'] for d in part_detections], axis=0) if len(part_detections)>0 else -1*np.ones((1,num_joints[part],2), dtype=np.float32) for part, part_detections in detections.items()}
            for k, v in det_poses2d.items():
                out_dir = os.path.join(output_dir, 'arrays', '2d', video_name)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                out_array = os.path.join(out_dir, f'{count}_{k}.npy')
                np.save(out_array, v[0])

        # save 3D results
        if save_3d:
            det_poses3d = {part: np.stack([d['pose3d'] for d in part_detections], axis=0) if len(part_detections)>0 else -1*np.ones((1,num_joints[part],2), dtype=np.float32) for part, part_detections in detections.items()}
            for k, v in det_poses3d.items():
                out_dir = os.path.join(output_dir, 'arrays', '3d', video_name)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                out_array = os.path.join(out_dir, f'{count}_{k}.npy')
                np.save(out_array, v[0])
      
        count += 1
        success, image = vidcap.read()
  
if __name__=="__main__":
    
    if torch.cuda.is_available():
        device = 'cuda:0'
        print('GPU available!')
    else:
        device = 'cpu'
        print('GPU not available... using CPU!')
    
    model_name='DOPE_v1_0_0'

    # load model
    ckpt_fname = os.path.join(_thisdir, 'models', model_name+'.pth.tgz')
    if not os.path.isfile(ckpt_fname):
        raise Exception('{:s} does not exist, please download the model first and place it in the models/ folder'.format(ckpt_fname))
    print('Loading model...')
    ckpt = torch.load(ckpt_fname, map_location=device)
    print(f'{model_name} successfully loaded!")
    #ckpt['half'] = False # uncomment this line in case your device cannot handle half computation
    ckpt['dope_kwargs']['rpn_post_nms_top_n_test'] = 1000
    model = dope_resnet50(**ckpt['dope_kwargs'])
    if ckpt['half']:
        model = model.half()
    model.eval()
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)
    
    toy_video_path = 'data/toy_video.mp4'
    dope_video(model, toy_video_path)
