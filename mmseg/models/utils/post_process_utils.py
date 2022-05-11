# import torch
# from torch.nn import functional as F
# import numpy as np

# try:
#     import pydensecrf.densecrf as dcrf
#     from pydensecrf.utils import (
#         unary_from_softmax,
#         unary_from_labels,
#         create_pairwise_bilateral,
#         create_pairwise_gaussian,
#     )
# except:
#     dcrf = None


# def dense_crf_post_process(
#     logits,
#     image,
#     n_labels=None,
#     max_iters=5,
#     pos_xy_std=(3, 3),
#     pos_w=3,
#     bi_xy_std=(80, 80),
#     bi_rgb_std=(13, 13, 13),
#     bi_w=10,
# ):
#     """
#     logits : [C,H,W]
#     image : [H,W,3]
#     """
#     if dcrf is None:
#         raise FileNotFoundError(
#             "pydensecrf is required to perform dense crf inference."
#         )
#     if isinstance(logits, torch.Tensor):
#         logits = F.softmax(logits, dim=0).detach().cpu().numpy()
#         U = unary_from_softmax(logits)
#         n_labels = logits.shape[0]
#     elif logits.ndim == 3:
#         U = unary_from_softmax(logits)
#         n_labels = logits.shape[0]
#     else:
#         assert n_labels is not None
#         U = unary_from_labels(logits, n_labels, zero_unsure=False)

#     d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], n_labels)

#     d.setUnaryEnergy(U)

#     # This adds the color-independent term, features are the locations only.
#     d.addPairwiseGaussian(
#         sxy=pos_xy_std,
#         compat=pos_w,
#         kernel=dcrf.DIAG_KERNEL,
#         normalization=dcrf.NORMALIZE_SYMMETRIC,
#     )

#     # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
#     d.addPairwiseBilateral(
#         sxy=bi_xy_std,
#         srgb=bi_rgb_std,
#         rgbim=image,
#         compat=bi_w,
#         kernel=dcrf.DIAG_KERNEL,
#         normalization=dcrf.NORMALIZE_SYMMETRIC,
#     )
#     # Run five inference steps.
#     logits = d.inference(max_iters)
#     logits = np.asarray(logits).reshape((n_labels, image.shape[0], image.shape[1]))
#     return torch.from_numpy(logits)


#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torch
import torch.nn.functional as F
# import torchvision.transforms.functional as VF
# from utils import unnorm

MAX_ITER = 10
POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3
BGR_MEAN = np.array([104.008, 116.669, 122.675])


def dense_crf_post_process(logits, image):
    # image = np.array(VF.to_pil_image(unnorm(image_tensor)))[:, :, ::-1]
    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)

    # output_logits = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear",
    #                               align_corners=False).squeeze()
    output_probs = F.softmax(logits, dim=0).cpu().numpy()

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return torch.from_numpy(Q)