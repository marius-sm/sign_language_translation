# Copyright 2020-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from collections import OrderedDict

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from collections import OrderedDict
import warnings
from typing import Tuple, List, Dict, Optional, Union

from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign

from torchvision.models import resnet
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform, resize_keypoints, resize_boxes

parts = ['body','hand','face']
num_joints = {'body': 13, 'hand': 21, 'face': 84}

class Dope_Transform(GeneralizedRCNNTransform):
  
    def __init__(self, min_size, max_size, image_mean, image_std):
        super(self.__class__, self).__init__(min_size, max_size, image_mean, image_std)

    def postprocess(self, result, image_shapes, original_image_sizes):
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            for k in ['pose2d', 'body_pose2d', 'hand_pose2d', 'face_pose2d']:
                if k in pred and pred[k] is not None:
                    pose2d = pred[k]
                    pose2d = resize_keypoints(pose2d, im_s, o_im_s)
                    result[i][k] = pose2d    
        return result


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.
    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(losses, detections, avg_features, max_features, max_indices):
        if self.training:
            return losses
        return detections, avg_features, max_features, max_indices

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        features, avg_features, max_features, max_indices = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return (losses, detections)
        else:
            # return self.eager_outputs(losses, detections, avg_features, max_features, max_indices)
            return detections, avg_features, max_features, max_indices


class Dope_RCNN(GeneralizedRCNN):

    def __init__(self, backbone,
                 dope_roi_pool, dope_head, dope_predictor,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # others
                 num_anchor_poses = {'body': 20, 'hand': 10, 'face': 10},
                 pose2d_reg_weights = {part: 5.0 for part in parts},
                 pose3d_reg_weights = {part: 5.0 for part in parts},
                ):
                
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(dope_roi_pool, (MultiScaleRoIAlign, type(None)))

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        dope_heads = Dope_RoIHeads(dope_roi_pool, dope_head, dope_predictor, num_anchor_poses, pose2d_reg_weights=pose2d_reg_weights, pose3d_reg_weights=pose3d_reg_weights)
            
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = Dope_Transform(min_size, max_size, image_mean, image_std)

        super(Dope_RCNN, self).__init__(backbone, rpn, dope_heads, transform)
        


class Dope_Predictor(nn.Module):

    def __init__(self, in_channels, dict_num_classes, dict_num_posereg):
        super(self.__class__, self).__init__()
        self.body_cls_score = nn.Linear(in_channels, dict_num_classes['body'])
        self.body_pose_pred = nn.Linear(in_channels, dict_num_posereg['body'])
        self.hand_cls_score = nn.Linear(in_channels, dict_num_classes['hand'])
        self.hand_pose_pred = nn.Linear(in_channels, dict_num_posereg['hand'])
        self.face_cls_score = nn.Linear(in_channels, dict_num_classes['face'])
        self.face_pose_pred = nn.Linear(in_channels, dict_num_posereg['face'])
        


    def forward(self, x):   
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = {}
        pose_deltas = {}
        scores['body'] = self.body_cls_score(x)
        pose_deltas['body'] = self.body_pose_pred(x)
        scores['hand'] = self.hand_cls_score(x)
        pose_deltas['hand'] = self.hand_pose_pred(x)
        scores['face'] = self.face_cls_score(x)
        pose_deltas['face'] = self.face_pose_pred(x)
        return scores, pose_deltas




class Dope_RoIHeads(RoIHeads):

    def __init__(self,
                 dope_roi_pool,
                 dope_head,
                 dope_predictor,
                 num_anchor_poses,
                 pose2d_reg_weights,
                 pose3d_reg_weights):
        
        fg_iou_thresh=0.5
        bg_iou_thresh=0.5
        batch_size_per_image=512
        positive_fraction=0.25
        bbox_reg_weights = [0.0]*4
        score_thresh = 0.0
        nms_thresh = 1.0
        detections_per_img = 99999999
        super(self.__class__, self).__init__(None, None, None, fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction, bbox_reg_weights,score_thresh,nms_thresh,detections_per_img,mask_roi_pool=None,mask_head=None,mask_predictor=None,keypoint_roi_pool=None,keypoint_head=None,keypoint_predictor=None)
        for k in parts:
            self.register_buffer(k+'_anchor_poses', torch.empty( (num_anchor_poses[k], num_joints[k], 5) ))
        self.dope_roi_pool = dope_roi_pool
        self.dope_head = dope_head
        self.dope_predictor = dope_predictor        
        self.J = num_joints
        self.pose2d_reg_weights = pose2d_reg_weights
        self.pose3d_reg_weights = pose3d_reg_weights

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[torch.Tensor])
            proposals (List[torch.Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        
        # roi_pool
        if features['0'].dtype==torch.float16: # UGLY: dope_roi_pool is not yet compatible with half
            features = {'0': features['0'].float()}
            if proposals[0].dtype==torch.float16:
                hproposals = [p.float() for p in proposals] 
            else:
                hproposals = proposals
            dope_features = self.dope_roi_pool(features, hproposals, image_shapes)
            dope_features = dope_features.half()          
        else:
            dope_features = self.dope_roi_pool(features, proposals, image_shapes)
            
        # head
        dope_features = self.dope_head(dope_features)
        
        # predictor
        class_logits, dope_regression = self.dope_predictor(dope_features)
        
        # process results
        result = []
        losses = {}
        if self.training:
            raise NotImplementedError
        else:
            boxes, scores, poses2d, poses3d = self.postprocess_dope(class_logits, dope_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                res = {'boxes': boxes[i]}
                for k in parts:
                  res[k+'_scores'] = scores[k][i]
                  res[k+'_pose2d'] = poses2d[k][i]
                  res[k+'_pose3d'] = poses3d[k][i]
                result.append(res)

        return result, losses
        
    def postprocess_dope(self, class_logits, dope_regression, proposals, image_shapes):
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        num_images = len(proposals)
        pred_scores = {}
        all_poses_2d = {}
        all_poses_3d = {}
        for k in parts:
            # anchor poses 
            anchor_poses = getattr(self, k+'_anchor_poses')
            nboxes, num_classes = class_logits[k].size()
            # scores
            sc = F.softmax(class_logits[k], -1)
            pred_scores[k] = sc.split(boxes_per_image, 0)
            # poses
            all_poses_2d[k] = []
            all_poses_3d[k] = []
            dope_regression[k] = dope_regression[k].view(nboxes, num_classes-1, self.J[k] * 5 )
            dope_regression_per_image = dope_regression[k].split(boxes_per_image, 0)
            for img_id in range(num_images):      
                dope_reg = dope_regression_per_image[img_id]
                boxes = proposals[img_id]
                # 2d
                offset = boxes[:,0:2]
                scale = boxes[:,2:4]-boxes[:,0:2]
                box_resized_anchors = offset[:,None,None,:] + anchor_poses[None,:,:,:2] * scale[:,None,None,:]
                dope_reg_2d = dope_reg[:,:,:2*self.J[k]].reshape(boxes.size(0),num_classes-1,self.J[k],2) / self.pose2d_reg_weights[k]
                pose2d = box_resized_anchors + dope_reg_2d * scale[:,None,None,:]
                all_poses_2d[k].append(pose2d)
                # 3d
                anchor3d = anchor_poses[None,:,:,-3:]
                dope_reg_3d = dope_reg[:,:,-3*self.J[k]:].reshape(boxes.size(0),num_classes-1,self.J[k],3) / self.pose3d_reg_weights[k]
                pose3d = anchor3d + dope_reg_3d
                all_poses_3d[k].append(pose3d)
        return proposals, pred_scores, all_poses_2d, all_poses_3d




def dope_resnet50(**dope_kwargs):

    backbone_name = 'resnet50'
    from torchvision.ops import misc as misc_nn_ops
    class FrozenBatchNorm2dWithHalf(misc_nn_ops.FrozenBatchNorm2d):    
        def forward(self, x):
            if x.dtype==torch.float16: # UGLY: seems that it does not work with half otherwise, so let's just use the standard bn function or half
                return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False)
            else:
                return super(self.__class__, self).forward(x)           
                
    backbone = resnet.__dict__[backbone_name](pretrained=False, norm_layer=FrozenBatchNorm2dWithHalf)
    # build the main blocks
    class ResNetBody(nn.Module):
        def __init__(self, backbone):
            super(self.__class__, self).__init__()
            self.resnet_backbone = backbone
            self.out_channels = 1024
            self.adaptmaxpool = nn.AdaptiveMaxPool2d(1, return_indices=True)
        def forward(self, x):
            x = self.resnet_backbone.conv1(x)
            x = self.resnet_backbone.bn1(x)
            x = self.resnet_backbone.relu(x)
            x = self.resnet_backbone.maxpool(x)
            x = self.resnet_backbone.layer1(x)
            x = self.resnet_backbone.layer2(x)
            x = self.resnet_backbone.layer3(x)
            y = self.resnet_backbone.avgpool(x)
            z, idxs = self.adaptmaxpool(x)
            return x, y.squeeze(), z.squeeze(), idxs.squeeze()
    resnet_body = ResNetBody(backbone)
    # build the anchor generator and pooler
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    # build the head and predictor
    class ResNetHead(nn.Module):
        def __init__(self, backbone):
            super(self.__class__, self).__init__()
            self.resnet_backbone = backbone
        def forward(self, x):
            x = self.resnet_backbone.layer4(x)
            x = self.resnet_backbone.avgpool(x)
            x = torch.flatten(x, 1)
            return x
    resnet_head = ResNetHead(backbone)
    
    # predictor
    num_anchor_poses = dope_kwargs['num_anchor_poses']
    num_classes = {k: v+1 for k,v in num_anchor_poses.items()}
    num_posereg =  {k: num_anchor_poses[k] * num_joints[k] * 5 for k in num_joints.keys()}
    predictor = Dope_Predictor(2048, num_classes, num_posereg)

    # full model 
    model = Dope_RCNN(resnet_body, roi_pooler, resnet_head, predictor, rpn_anchor_generator=anchor_generator, **dope_kwargs)

    return model
