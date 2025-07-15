import torch
import logging
import numpy as np
from torch import nn
from typing import Dict
from pytorch_metric_learning import losses
from detectron2.utils import comm
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.utils.events import get_event_storage
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from defrcn.dataloader import MetadataCatalog
from .box_head import build_box_head
from .fast_rcnn import ROI_HEADS_OUTPUT_REGISTRY, FastRCNNOutputLayers, FastRCNNOutputs

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        )

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes,
            self.batch_size_per_image,
            self.positive_sample_fraction,
            self.num_classes,
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                Other fields such as "gt_classes" that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix
            )
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # will filter the proposals again (by foreground/background,
                # etc), so we essentially index the data twice.
                for (
                    trg_name,
                    trg_value,
                ) in targets_per_image.get_fields().items():
                    if trg_name.startswith(
                        "gt_"
                    ) and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(
                            trg_name, trg_value[sampled_targets]
                        )
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros(
                        (len(sampled_idxs), 4)
                    )
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append(
                (gt_classes == self.num_classes).sum().item()
            )
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
                detected instances. Returned during inference only; may be []
                during training.
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where the heads share the
    cropping and the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg, out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            stride_per_block=[2, 1, 1],
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        # print('pooler:', x.size())
        x = self.res5(x)
        # print('res5:', x.size())
        return x

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled
        )
        del feature_pooled

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if self.training:
            del features
            losses = outputs.losses()
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )

        self.cls_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )

        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes,
            self.cls_agnostic_bbox_reg,
        )

        self.cls_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes,
            self.cls_agnostic_bbox_reg,
        )

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]

        if self.training:
            losses = self._forward_box(features_list, proposals)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            return pred_instances, {}

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )

        cls_features = self.cls_head(box_features)
        pred_class_logits, _ = self.cls_predictor(
            cls_features
        )

        box_features = self.box_head(box_features)
        _, pred_proposal_deltas = self.box_predictor(
            box_features
        )
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances


import copy
import wandb
import torch.distributed as dist
from torch import distributions
import torch.nn.functional as F
from detectron2.layers import cat
import fvcore.nn.weight_init as weight_init
from geom_median.torch import geometric_median_tensor

@ROI_HEADS_REGISTRY.register()
class CommonalityROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where the heads share the
    cropping and the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        
        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg, out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        )
        
        self.fc_s = nn.Linear(out_channels, out_channels)
        self.fc_l = nn.Linear(out_channels, out_channels)
        
        for layer in [self.fc_s, self.fc_l]:
            weight_init.c2_xavier_fill(layer)
     
        self.cfg = cfg
        self.device = cfg.MODEL.DEVICE
        self.memory = cfg.MODEL.ROI_HEADS.MEMORY
        self.semantic = cfg.MODEL.ROI_HEADS.SEMANTIC
        self.augmentation = cfg.MODEL.ROI_HEADS.AUGMENTATION
        self.warmup_distill = cfg.MODEL.ROI_HEADS.WARMUP_DISTILL
        self.top_k = 2
        self.momentum = 0.999
        self.exp_dist = torch.distributions.exponential.Exponential(torch.tensor([2.75]))
        self.sup_con_loss = losses.SupConLoss()
        # create the queue
        self.queue_len = cfg.MODEL.ROI_HEADS.QUEUE_LEN
        self._moco_encoder_init(self.cfg)
        if self.memory:
            self.register_buffer("queue_s", torch.zeros(self.num_classes, self.queue_len, out_channels))
            self.register_buffer("queue_l", torch.zeros(self.num_classes, self.queue_len, out_channels))
            self.register_buffer("queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))
            self.register_buffer("queue_full", torch.zeros(self.num_classes, dtype=torch.long))
        self.class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes
        if self.num_classes in [15, 20]:
            self.novel_index = [15, 16, 17, 18, 19]
        elif self.num_classes in [60, 80]:
            self.novel_index = [0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62]
        self.base_index = []
        for i in range(self.num_classes):
            if i not in self.novel_index:
                self.base_index.append(i)
    
    def _moco_encoder_init(self, cfg, *args):
        # ROI Box Head
        #for param_q, param_k in zip(self.box_head_q.parameters(),
        #                            self.box_head_k.parameters()):
        #    param_k.data.copy_(param_q.data)
        #    param_k.requires_grad = False

        # MLP head
        self.mlp_q = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128),
        )
        for layer in self.mlp_q:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)
        self.mlp_k = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128),
        )
        for param_q, param_k in zip(self.mlp_q.parameters(),
                                    self.mlp_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update(self):
        #for param_q, param_k in zip(self.box_head_q.parameters(),
        #                            self.box_head_k.parameters()):
        #    param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)

        for param_q, param_k in zip(self.mlp_q.parameters(),
                                    self.mlp_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)
    
    @torch.no_grad()
    def _get_class_names(self, pred_classes):
        pred_class_names = list(map(lambda x: self.class_names[x], pred_classes))
        return pred_class_names

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_s, keys_l, gt_class):
        keys_s = keys_s[:self.queue_len]
        keys_l = keys_l[:self.queue_len]
        #num_samples = keys_s.shape[0] //4
        #mean = torch.zeros(2048)
        #cov = 5*torch.ones(num_samples, 2048)
        #keys_s[:num_samples, :] = keys_s[:num_samples, :] + torch.normal(mean, cov).to(keys_s.device) 
        #keys_l[:num_samples, :] = keys_l[:num_samples, :] + torch.normal(mean, cov).to(keys_s.device)
        batch_size = keys_s.shape[0]
        ptr = int(self.queue_ptr[gt_class])
        if ptr + batch_size <= self.queue_len:
            self.queue_s[gt_class, ptr:ptr + batch_size] = keys_s
            self.queue_l[gt_class, ptr:ptr + batch_size] = keys_l
        else:
            self.queue_s[gt_class, ptr:] = keys_s[:self.queue_len - ptr]
            self.queue_s[gt_class, :(ptr + batch_size) % self.queue_len] = keys_s[self.queue_len - ptr:]
            self.queue_l[gt_class, ptr:] = keys_l[:self.queue_len - ptr]
            self.queue_l[gt_class, :(ptr + batch_size) % self.queue_len] = keys_l[self.queue_len - ptr:]
            
        if ptr + batch_size >= self.queue_len:
            self.queue_full[gt_class] = 1
        ptr = (ptr + batch_size) % self.queue_len
        self.queue_ptr[gt_class] = ptr
    
    @torch.no_grad()
    def update_memory(self, features_s, features_l, gt_classes):
        features_s = concat_all_gather(features_s)
        features_l = concat_all_gather(features_l)
        gt_classes = concat_all_gather(gt_classes)
        
        fg_cases = (gt_classes >= 0) & (gt_classes < self.num_classes)
        features_fg_s = features_s[fg_cases]
        features_fg_l = features_l[fg_cases]
        gt_classes_fg = gt_classes[fg_cases]
        
        if len(gt_classes_fg) == 0:
            return
        uniq_c = torch.unique(gt_classes_fg)
            
        for c in uniq_c:
            c = int(c)
            c_index = torch.nonzero(
                gt_classes_fg == c, as_tuple=False
            ).squeeze(1)
            features_c_s = features_fg_s[c_index]
            features_c_l = features_fg_l[c_index]
            self._dequeue_and_enqueue(features_c_s, features_c_l, c)

    @torch.no_grad()
    def predict_prototype(self, feature_pooled_s, feature_pooled_l, gt_classes):
        
        prototypes_s = []
        prototypes_l = []
        for i in range(self.num_classes):
            if self.queue_full[i] or self.queue_ptr[i] == 0:
                #logger.info('self.queue_s.shape: {}'.format(self.queue_s[i].mean(dim=0).shape))
                #logger.info('self.queue_l.shape: {}'.format(self.queue_l[i].shape))
                s_mean, l_mean = self.queue_s[i].mean(dim=0), self.queue_l[i].mean(dim=0)
                #prot_s, prot_l = self.queue_s[i], self.queue_l[i]
                #weights_s, weights_l = torch.ones(len(prot_s), device=self.device), torch.ones(len(prot_l), device=self.device)
                #with torch.no_grad():
                #    out_s = geometric_median_tensor(prot_s, weights_s, eps=1e-6, maxiter=100, ftol=1e-10)
                #    out_l = geometric_median_tensor(prot_l, weights_l, eps=1e-6, maxiter=100, ftol=1e-10)
                #if comm.is_main_process():
                #    wandb.log({'prot_s': torch.linalg.norm(out_s.median - s_mean), 'prot_l': torch.linalg.norm(out_l.median - l_mean)})
                prototypes_s.append(s_mean)
                prototypes_l.append(l_mean)
                #prototypes_s.append(out_s.median)
                #prototypes_l.append(out_l.median)
            else:
                #logger.info('self.queue_s.shape: {}'.format(self.queue_s[i][:self.queue_ptr[i]].mean(dim=0).shape))
                s_mean, l_mean = self.queue_s[i][:self.queue_ptr[i]].mean(dim=0), self.queue_l[i][:self.queue_ptr[i]].mean(dim=0)
                #prot_s, prot_l = self.queue_s[i][:self.queue_ptr[i]], self.queue_l[i][:self.queue_ptr[i]]
                #weights_s, weights_l = torch.ones(len(prot_s), device=self.device), torch.ones(len(prot_l), device=self.device)
                #with torch.no_grad():
                #    out_s = geometric_median_tensor(prot_s, weights_s, eps=1e-6, maxiter=100, ftol=1e-10)
                #    out_l = geometric_median_tensor(prot_l, weights_l, eps=1e-6, maxiter=100, ftol=1e-10)
                #if comm.is_main_process():
                #    wandb.log({'prot_s': torch.linalg.norm(out_s.median - s_mean), 'prot_l': torch.linalg.norm(out_l.median - l_mean)})
                prototypes_s.append(s_mean)
                prototypes_l.append(l_mean)
                #prototypes_s.append(out_s.median)
                #prototypes_l.append(out_l.median)
                
                    
        prototypes_s = torch.stack(prototypes_s, dim=0)
        prototypes_l = torch.stack(prototypes_l, dim=0)
        
        predict_cosine_s = F.cosine_similarity(feature_pooled_s[:, None], prototypes_s[None, :], dim=-1)
        predict_classes_s = predict_cosine_s.new_full(predict_cosine_s.size(), -1.0)
        predict_classes_s[:, self.novel_index] = predict_cosine_s[:, self.novel_index]
        
        predict_cosine_l = F.cosine_similarity(feature_pooled_l[:, None], prototypes_l[None, :], dim=-1)
        predict_classes_l = predict_cosine_l.new_full(predict_cosine_l.size(), -1.0)
        predict_classes_l[:, self.novel_index] = predict_cosine_l[:, self.novel_index]
        
        for i in range(len(predict_classes_s)):
            if gt_classes[i] == self.num_classes or gt_classes[i] < 0:
                continue
            predict_classes_s[i, gt_classes[i]] = predict_cosine_s[i, gt_classes[i]]
            predict_classes_l[i, gt_classes[i]] = predict_cosine_l[i, gt_classes[i]]
        
        zeros = torch.zeros((len(predict_classes_s), 1), device=predict_classes_s.device)
        
        predict_classes_s = F.softmax(predict_classes_s*10, dim=-1)
        predict_classes_s = torch.cat([predict_classes_s, zeros], dim=1) #For the background class
        
        predict_classes_l = F.softmax(predict_classes_l*10, dim=-1)
        predict_classes_l = torch.cat([predict_classes_l, zeros], dim=1)
        
        return predict_classes_s, predict_classes_l
    
    @torch.no_grad()
    def generate_features(self, gt_classes):
        new_features = []
        new_classes = []
        uniq_c = torch.unique(gt_classes)
        kth = 2
        num_samples = 10
        base_features = self.queue_s[self.base_index]

        # base_mean = base_features.mean(dim=1)
        # base_std = base_features.var(dim=0, unbiased=False)
        # base_std = base_std * self.queue_len / (self.queue_len - 1)

        ####
        base_mean, base_std = [], []
        for c in self.base_index:
            flag = self.queue_full[c]
            if self.queue_full[c]:
                #print('Line692; self.queue_s.shape: {}'.format(self.queue_s[c].shape))
                c_mean = self.queue_s[c].mean(dim=0)
                #prot_s = self.queue_s[c]
                #weights_s = torch.ones(len(prot_s), device=self.device)
                #with torch.no_grad():
                #    out_s = geometric_median_tensor(prot_s, weights_s, eps=1e-6, maxiter=100, ftol=1e-10)
                #
                #if comm.is_main_process():
                #    wandb.log({'out_s': torch.linalg.norm(c_mean - out_s.median)})
                #c_mean = out_s.median
                c_std = self.queue_s[c].var(dim=0, unbiased=False)
                #print('c_std {}'.format(c_std))
                c_std = c_std * self.queue_len / (self.queue_len - 1)
                
            else:
                c_mean = self.queue_s[c][:self.queue_ptr[c]].mean(dim=0)
                #prot_s = self.queue_s[c][:self.queue_ptr[c]]
                #weights_s = torch.ones(len(prot_s), device=self.device)
                #with torch.no_grad():
                #    out_s = geometric_median_tensor(prot_s, weights_s, eps=1e-6, maxiter=100, ftol=1e-10)
                #if comm.is_main_process():
                #    wandb.log({'out_s': torch.linalg.norm(c_mean - out_s.median)})
                #c_mean = out_s.median
                c_std = self.queue_s[c][:self.queue_ptr[c]].var(dim=0, unbiased=False)
                #print('c_std {}'.format(c_std))
                if self.queue_ptr[c] > 1:
                    c_std = c_std * self.queue_ptr[c] / (self.queue_ptr[c] - 1)
                
            
            #logger.info('c_std {}'.format(torch.all(c_std > 0)))
            #logger.info('Number of Zeros: {}'.format(torch.sum(c_std ==0)))
            c_std[c_std == 0] = 0.01
            #if torch.any(c_std < 0) or torch.any(c_std == 0):
            #    torch.save(c_std, 'c_std.pt')
            #    logger.warning('c_std is {}'.format(c_std))
            #    logger.info('c_std_calc: {}'.format() )
            #if not torch.all(c_std > 0):
            #    logger.info('FLAG: {}'.format(flag))
            #    if flag:
            #        full = self.queue_s[c].var(dim=0, unbiased=False)
            #        torch.save(full, 'c_std_full.pt')
            #        queue_len_full = self.queue_len - 1
            #        torch.save(queue_len_full, 'queue_len_full.pt')
            #    else:                
            #        part = self.queue_s[c][:self.queue_ptr[c]].var(dim=0, unbiased=False)
            #        torch.save(part, 'c_std_part.pt')
            #        queue_len_part = (self.queue_ptr[c] - 1)
            #        torch.save(queue_len_part, 'queue_len_part.pt')
            #assert torch.all(c_std > 0)
            if torch.sum(torch.isnan(c_std)) > 0:
                continue
            base_mean.append(c_mean)
            base_std.append(c_std)
        base_mean = torch.stack(base_mean, dim=0)
        base_std = torch.stack(base_std, dim=0)
        ####

        for c in self.novel_index:
            if np.random.rand() < 0.7 or (c not in uniq_c):
                continue
            if self.queue_full[c]:
                c_mean = self.queue_s[c].mean(dim=0)
                #prot_s = self.queue_s[c][:self.queue_ptr[c]]
                #weights_s = torch.ones(len(prot_s), device=self.device)
                #with torch.no_grad():
                #    out_s = geometric_median_tensor(prot_s, weights_s, eps=1e-6, maxiter=100, ftol=1e-10)
                #if comm.is_main_process():
                #    wandb.log({'out_s': torch.linalg.norm(c_mean - out_s.median)})
                #c_mean = out_s.median
                c_std = self.queue_s[c].var(dim=0, unbiased=False)
                c_std = c_std * self.queue_len / (self.queue_len - 1)
            else:
                c_mean = self.queue_s[c][:self.queue_ptr[c]].mean(dim=0)
                #prot_s = self.queue_s[c][:self.queue_ptr[c]]
                #weights_s = torch.ones(len(prot_s), device=self.device)
                #with torch.no_grad():
                #    out_s = geometric_median_tensor(prot_s, weights_s, eps=1e-6, maxiter=100, ftol=1e-10)
                #if comm.is_main_process():
                #    wandb.log({'out_s': torch.linalg.norm(c_mean - out_s.median)})
                #c_mean = out_s.median
                c_std = self.queue_s[c][:self.queue_ptr[c]].var(dim=0, unbiased=False)
                if self.queue_ptr[c] > 1:
                    c_std = c_std * self.queue_ptr[c] / (self.queue_ptr[c] - 1)
            
            
            #dists = self.mahalanobis_distance(c_mean, base_mean, base_std)
            dists = torch.norm(c_mean[None,:] - base_mean, p=2, dim=1)
            _, index = dists.sort()
            #mean = torch.cat([self.features_mean[c].unsqueeze(0), self.features_mean[index[:kth]]])
            calibrated_mean = c_mean
            calibrated_std = base_std[index[:kth]].mean(dim=0)
            
            univariate_normal_dists = distributions.normal.Normal(
                calibrated_mean, scale=torch.sqrt(calibrated_std))
            
            feaures_rsample = univariate_normal_dists.rsample(
                (num_samples,))
            classes_rsample = gt_classes.new_full((num_samples, ), c)
            
            new_features.append(feaures_rsample)
            new_classes.append(classes_rsample)
        if len(new_features) == 0:
            return [], []
        else:
            return torch.cat(new_features), torch.cat(new_classes)
    
    def mahalanobis_distance(self, c_mean, base_mean, base_std):
        inv_row = 1 / base_std
        inv = torch.diag_embed(inv_row)
        #inv = torch.linalg.inv(cov_mats)

        diff  = torch.unsqueeze(c_mean - base_mean, -1)
        diff_transpose = torch.permute(diff, (0,2,1))
        
        first = torch.matmul(inv, diff)
        dist = torch.matmul(diff_transpose, first)

        dist = torch.squeeze(dist)

        return dist
    
    @torch.no_grad()
    def get_memory_features(self):
        MAX = 25
        class_idx = self.base_index + self.novel_index
        mem_features = torch.ones(1, 2048).to(self.device)
        mem_labels = torch.ones(1).to(self.device)
        for c in class_idx:
            if self.queue_full[c]:
                prot_s = self.queue_s[c].to(self.device)
            else:
                prot_s = self.queue_s[c][:self.queue_ptr[c]]

            if len(prot_s) > 0:
                mem_features = torch.vstack((mem_features, prot_s))
                labels = c * torch.ones(len(prot_s)).to(self.device)
                mem_labels = torch.cat((mem_labels, labels))

        mem_features = mem_features[1:, :]
        mem_features = F.normalize(mem_features, dim=1).unsqueeze(0)
        mem_labels = mem_labels[1:]
        
        return mem_features, mem_labels
    
    @torch.no_grad()
    def set_difference(self, t1, t2):
        combined = torch.cat((t1, t2))
        uniques, counts = combined.unique(return_counts=True)
        difference = uniques[counts == 1]
        intersection = uniques[counts > 1]

        return difference

    @torch.no_grad()
    def get_memory_features_weighted(self, foreground_embed, foreground_label, values, indices):
        MAX = 200
        TOTAL = 2000
        N,M = values.shape
        num_foreground = foreground_embed.shape[0]
        foreground_feat = foreground_embed.clone().detach().unsqueeze(1)
        mem_features, mem_labels = self.get_memory_features()
        mem_features = self.mlp_k(mem_features)
        mem_features = F.normalize(mem_features, dim=2)
        all_idx = torch.arange(len(mem_features[0]), device=self.device)
        features, labels = [], []
        for i in range(0, num_foreground):

            forg_feat, forg_label = foreground_feat[i, :], foreground_label[i]
            sim = torch.sum(forg_feat * mem_features, dim=2)
            sim_mining, sort_index = torch.sort(sim)
            sim_mining_features = mem_features[0, sort_index, :]
            sim_mining_labels = mem_labels[sort_index]
            
            #Get all foreground features as posivies
            forg_class_idx = torch.where(sim_mining_labels[0,:] == forg_label)[0]
            other_forg_class_features = sim_mining_features[0:1, forg_class_idx, :]
            other_forg_class_labels = sim_mining_labels[0:1, forg_class_idx]
            other_forg_class_sim = sim_mining[0:1, forg_class_idx]

            final_foreground_features = other_forg_class_features[0:1, 0:MAX, :]
            final_foreground_label = other_forg_class_labels[0:1, 0:MAX]
            
            closest = indices[i, 1] if indices[i, 0] == forg_label else indices[i, 0]
            closest_class_idx = torch.where(sim_mining_labels[0,:] == closest)[0]
            closest_class_features = sim_mining_features[0:1, closest_class_idx, :]
            closest_class_labels = sim_mining_labels[0:1, forg_class_idx]
            closest_class_sim = sim_mining[0:1, closest_class_idx]
            
            random_weights = self.exp_dist.sample(closest_class_sim.shape).to(self.device)
            random_weights = random_weights * (random_weights < 1)
            random_weights = 1 - random_weights
            #Mix-up features
            sim_weighted_forg_feat = random_weights.T * forg_feat
            sim_weighted_closes_class_features = (1 - random_weights.T)  * closest_class_features
            new_features = sim_weighted_forg_feat + sim_weighted_closes_class_features
            new_features = new_features[0:1, -MAX:, :]
            new_labels = forg_label * torch.ones(1, MAX, dtype=torch.float32, device=self.device)

            ##Negatives
            #most_negative = sim_mining_features[0:1, -MAX:, :]
            #most_negative_sim  = sim_mining[0:1, -MAX:]
            #negative_weighted_forg_feat = most_negative_sim.T * forg_feat
            #negative_weighted_closes_class_features =  (1 - most_negative_sim.T) * closest_class_features
            #new_features = sim_weighted_forg_feat + sim_weighted_closes_class_features
            
            #Remove the selected features from the memory bank set
            idx_to_remove = torch.cat((forg_class_idx, closest_class_idx[-MAX:]))
            idx_to_keep  = self.set_difference(all_idx, idx_to_remove)

            remaining_feat = sim_mining_features[0:1, idx_to_keep, :]
            remaining_labels = sim_mining_labels[0:1, idx_to_keep]
            remaining_sim = sim_mining[0:1, idx_to_keep]

            num_positive_feat = final_foreground_features.shape[1] + new_features.shape[1]
            # Get the hardest remaining negatives
            final_remaining_feat = remaining_feat[0:1, -(TOTAL - num_positive_feat):, :]
            final_remaining_label = remaining_labels[0:1, -(TOTAL - num_positive_feat):]
            final_remaining_sim = remaining_sim[0:1, -(TOTAL - num_positive_feat):]

            ##Negatives
            
            neg_augment_feat = remaining_feat[0:1, -MAX:, :]
            neg_augment_label = remaining_labels[0:1, -MAX:]
            neg_augment_sim = remaining_sim[0:1, -MAX:]
            neg_forg_feat = neg_augment_sim.T * forg_feat
            neg_closes_class_features =  (1 - neg_augment_sim.T) * neg_augment_feat
            neg_features = neg_forg_feat + neg_closes_class_features
            
            
            
            final_feat = torch.cat((final_foreground_features, new_features, final_remaining_feat, neg_features), dim=1)
            final_label = torch.cat((final_foreground_label, new_labels, final_remaining_label, neg_augment_label), dim=1)

            #final_feat = torch.cat((final_foreground_features, final_remaining_feat), dim=1)
            #final_label = torch.cat((final_foreground_label, final_remaining_label), dim=1)

            #if final_feat.shape[1] < TOTAL:
            #    temp = 5

            features.append(final_feat)
            labels.append(final_label)
            
        
        augmented_mem_bank_features = torch.stack(features)[:, 0, :]        
        augmented_mem_bank_labels = torch.stack(labels)[:, 0, :]
        
        augmented_mem_bank_features = F.normalize(augmented_mem_bank_features, dim=2).detach()
        
        return augmented_mem_bank_features, augmented_mem_bank_labels
    
    def score_other_classes(self, foreground_labels, softmax_scores, mask):
        
        N = len(softmax_scores)
        weights = torch.ones(mask.shape, device=self.device)
        values, indices = torch.topk(softmax_scores, self.top_k, dim=1)
        

        return weights, mask
    
    @torch.no_grad()
    def compute_key_featuers(self, k_box_features):
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update() # update key encoder
            k_embedding = self.mlp_k(k_box_features)
            k = F.normalize(k_embedding)
        return k
    
    def other_compute_supcon_loss(self, embeddings, labels, softmax_scores):
        
        temperature = 0.07
        foreground_idx = labels < self.num_classes
        foreground_labels = labels[foreground_idx]
        foreground_embeddings = embeddings[foreground_idx, :]
        #Project to queries
        foreground_embeddings = self.mlp_q(foreground_embeddings)
        foreground_embeddings = F.normalize(foreground_embeddings, dim=1)
        N = len(foreground_embeddings)
        slice_foreground_label = foreground_labels.unsqueeze(1)     
        
        novel_idx_tensor =  torch.tensor(self.novel_index).to(self.device)
        out = torch.eq(slice_foreground_label, novel_idx_tensor)
        novel_keep_idx = torch.any(out, dim=1)
        randidx = torch.randperm(N)[0:(N//4)]
        foreground_embeddings, foreground_labels, softmax_scores = foreground_embeddings[novel_keep_idx], foreground_labels[novel_keep_idx], softmax_scores[novel_keep_idx, :]
        assert len(foreground_embeddings) == len(foreground_labels) == len(softmax_scores)
 
        
        value, indices = torch.topk(softmax_scores, self.top_k, dim=1)
        #weights, mask = self.score_other_classes(foreground_labels, softmax_scores, mask)
        
        #Get the embeddings from the memory bank
        with torch.no_grad():
            self._momentum_update()
            mem_features, mem_labels = self.get_memory_features_weighted(foreground_embeddings, foreground_labels, value, indices)
                
        #
        foreground_samples, mem_bank_samples = mem_labels.shape
        foreground_augment, labels_augment = foreground_embeddings.unsqueeze(1), foreground_labels.unsqueeze(1)
        labels_augment = torch.repeat_interleave(labels_augment, mem_bank_samples, dim=1)
        mask = torch.eq(labels_augment, mem_labels)
        
        
        sim =  torch.div(torch.sum(foreground_augment*mem_features, dim=2), temperature)        
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        #
        loss = -1 * mean_log_prob_pos
        
        loss = loss.mean()

        return loss
    ''' 
    def compute_supcon_loss(self, embeddings, labels, softmax_scores):
        
        temperature = 0.07
        foreground_idx = labels < self.num_classes
        foreground_labels = labels[foreground_idx]
        foreground_embeddings = embeddings[foreground_idx, :]
        assert len(foreground_embeddings) == len(foreground_labels) == len(softmax_scores)

        foreground_embeddings = F.normalize(foreground_embeddings, dim=1)
        class_idx = torch.unique(foreground_labels)      
        
        #Get the embeddings from the memory bank
        mem_features, mem_labels = self.get_memory_features(class_idx)
        mem_features = F.normalize(mem_features, p=2, dim=1).detach()

        full_embeddings = torch.cat((foreground_embeddings, mem_features))
        full_labels = torch.cat((foreground_labels, mem_labels))

        #Mask for foreground labels
        mask = torch.eq(full_labels.view(-1,1), full_labels.view(-1, 1).T)
        
        
        #Check for valid boxes (i.e. if there are no positives, don't use that proposal)
        valid_boxes = torch.sum(mask, dim=1) > 1
        mask = mask[:, valid_boxes]
        mask = mask[valid_boxes, :]
        full_labels = full_labels[valid_boxes]    
        full_embeddings = full_embeddings[valid_boxes, :]

        weights, mask = self.score_other_classes(foreground_labels, softmax_scores, mask)
        one_mat = torch.ones_like(mask, dtype=torch.int, device=self.device)
        diag = torch.arange(len(full_labels), device=self.device).view(-1,1)
        logits_mask = torch.scatter(one_mat, 1, diag, 0)
        mask = mask * logits_mask
        
        #Similarity computation
        sim = torch.div(torch.matmul(full_embeddings, full_embeddings.T), temperature)
        
        assert torch.sum(torch.isnan(sim)) == 0
        assert len(weights) == len(sim)
        #Logits computation
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob * weights).sum(1) / mask.sum(1)


        loss = -1 * mean_log_prob_pos
        
        loss = loss.mean()

        return loss
    
    '''
        
    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            stride_per_block=[2, 1, 1],
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        # print('pooler:', x.size())
        x = self.res5(x)
        # print('res5:', x.size())
        return x

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        
        del images
        #logger.info('Inside forward')
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        
        feature_pooled_s = F.relu(self.fc_s(feature_pooled))
        feature_pooled_l = F.relu(self.fc_l(feature_pooled))
        
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled_s, feature_pooled_l
        )

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if self.training:          
            
            if self.memory:
                with torch.no_grad():
                    gt_classes = outputs.gt_classes
                    pad_size = self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE \
                        * self.cfg.SOLVER.IMS_PER_BATCH // torch.distributed.get_world_size()
                    if self.cfg.DATASETS.TWO_STREAM:
                        pad_size *= 2
                    feature_pooled_pad_s = feature_pooled_s.new_full((
                        pad_size, feature_pooled_s.size(1)), -1)
                    feature_pooled_pad_s[: feature_pooled_s.size(0)] = feature_pooled_s
                    feature_pooled_pad_l = feature_pooled_l.new_full((
                        pad_size, feature_pooled_l.size(1)), -1)
                    feature_pooled_pad_l[: feature_pooled_l.size(0)] = feature_pooled_l
                    gt_classes_pad = gt_classes.new_full((pad_size,), -1)
                    gt_classes_pad[: gt_classes.size(0)] = gt_classes
                    
                    self.update_memory(feature_pooled_pad_s.detach(), feature_pooled_pad_l.detach(), gt_classes_pad)

            losses = outputs.losses() 
            
            #if int(storage.iter) <= self.warmup_distill:
            #    gt_classes = outputs.gt_classes
            #    bg_class_ind = pred_class_logits.shape[1] - 1
            #    true_cases = (gt_classes >= 0) & (gt_classes < bg_class_ind)
            #    
            #    gt_prototype_classes_s,  gt_prototype_classes_l= self.predict_prototype(feature_pooled_s, feature_pooled_l, gt_classes)
            #    gt_prototype_classes_s = gt_prototype_classes_s.detach()
            #    supconloss = self.other_compute_supcon_loss(feature_pooled_s, gt_classes, gt_prototype_classes_s[true_cases])                
            #    losses['supconloss'] = 0.05 * supconloss
            storage = get_event_storage()
            if int(storage.iter) >= self.warmup_distill:
                
                
                if self.semantic:
                    #logger.info('Inside self.semantic')
                    gt_classes = outputs.gt_classes
                    bg_class_ind = pred_class_logits.shape[1] - 1
                    true_cases = (gt_classes >= 0) & (gt_classes < bg_class_ind)
                    
                    gt_prototype_classes_s,  gt_prototype_classes_l= self.predict_prototype(feature_pooled_s, feature_pooled_l, gt_classes)
                    gt_prototype_classes_s = gt_prototype_classes_s.detach()
                    gt_prototype_classes_l = gt_prototype_classes_l.detach()
                
                    losses = outputs.losses()
                    loss_kld = F.kl_div(F.log_softmax(pred_class_logits[true_cases], dim=1),
                        gt_prototype_classes_s[true_cases], reduction='batchmean')
                    loss_reg_disitll = outputs.smooth_l1_loss_distill(gt_prototype_classes_l)
                    losses.update({'loss_kld': loss_kld * 0.1, 'loss_reg_disitll': loss_reg_disitll * 0.7})
                    losses['loss_cls'] = losses['loss_cls'] * 1.0
                    losses['loss_box_reg'] = losses['loss_box_reg'] * 0.7

                    supconloss = self.other_compute_supcon_loss(feature_pooled_s, gt_classes, gt_prototype_classes_s[true_cases])                
                    losses['supconloss'] = 0.10 * supconloss
                    
                if self.augmentation:
                    new_features, new_classes = self.generate_features(gt_classes)
                    if len(new_features) == 0:
                        loss_cls_score_aug = feature_pooled_pad_s.new_full((1,), 0).mean()
                    else:
                        pred_class_logits_aug, _ = self.box_predictor(
                            new_features, new_features
                        )
                        loss_cls_score_aug = F.cross_entropy(
                            pred_class_logits_aug, new_classes, reduction="mean"
                        )

                    losses.update({"loss_cls_score_aug": loss_cls_score_aug * 0.1})

            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances, {}


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output

