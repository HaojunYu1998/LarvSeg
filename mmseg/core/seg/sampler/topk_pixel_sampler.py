# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from ..builder import PIXEL_SAMPLERS
from .base_pixel_sampler import BasePixelSampler


@PIXEL_SAMPLERS.register_module()
class TopKPixelSampler(BasePixelSampler):
    """Online Hard Example Mining Sampler for segmentation.

    Args:
        context (nn.Module): The context of sampler, subclass of
            :obj:`BaseDecodeHead`.
        thresh (float, optional): The threshold for hard example selection.
            Below which, are prediction with low confidence. If not
            specified, the hard examples will be pixels of top ``min_kept``
            loss. Default: None.
        min_kept (int, optional): The minimum number of predictions to keep.
            Default: 100000.
    """

    def __init__(self, context, sample_rate=0.1, min_kept=10):
        super(TopKPixelSampler, self).__init__()
        self.context = context
        self.sample_rate = sample_rate
        assert min_kept > 1
        self.min_kept = min_kept

    def sample(self, seg_logit, seg_label):
        """Sample pixels that have high loss or with low prediction confidence.

        Args:
            seg_logit (torch.Tensor): segmentation logits, shape (N, C, H, W)
            seg_label (torch.Tensor): segmentation label, shape (N, 1, H, W)

        Returns:
            torch.Tensor: segmentation weight, shape (N, H, W)
        """
        # with torch.no_grad():
        # gt_semantic_seg: B, 1, H, W
        # masks: B, N, H, W
        seg_logit = F.interpolate(
            seg_logit, size=seg_label.shape[-2:], mode="bilinear", align_corners=False
        )
        B, N, H, W = seg_logit.size()
        assert B == 1, "Only support batch == 1 for segmenter!"
        seg_label = F.interpolate(seg_label.float(), size=(H, W)).long()
        seg_label = seg_label.reshape(B * H * W)
        unique_label = torch.unique(seg_label)
        unique_label = unique_label[unique_label != self.context.ignore_index]
        pos_bucket = [
            torch.nonzero(seg_label == l)[:, 0]
            for l in unique_label
        ]
        if len(pos_bucket) == 0:
            return [], []
        seg_logit = seg_logit.permute(0, 2, 3, 1).reshape(B * H * W, N)
        prior_buckets = self._sample(seg_logit, pos_bucket, unique_label)
        return pos_bucket, prior_buckets

    def _sample(self, seg_logit, buckets, unique_label, sample_rate=0.1):
        """Sample points from each buckets
        Args:
            num_per_buckets (list): number of points in each class
        """
        num_per_buckets = []
        for p in buckets:
            k = int(sample_rate * len(p))
            if k < self.min_kept:
                k = min(self.min_kept, len(p))
            num_per_buckets.append(k)
        prior_buckets = []
        for k, p, l in zip(num_per_buckets, buckets, unique_label):
            prior_buckets.append(
                p[seg_logit[p, int(l)].topk(k).indices]
            )
        return prior_buckets