import math

import torch

import torchvision
from torchvision.ops import roi_align as torchvision_roi_align
from torchvision.ops.boxes import box_area


def roi_align(features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio):
    if torch.__version__ >= "1.5.0":
        return torch.ops.torchvision.roi_align(
            features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, False)
    else:
        return torch.ops.torchvision.roi_align(
            features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio)


class RoIAlign:
    """
    Performs Region of Interest (RoI) Align operator described in Mask R-CNN
    
    """
    
    def __init__(self, output_size, sampling_ratio):
        """
        Arguments:
            output_size (Tuple[int, int]): the size of the output after the cropping
                is performed, as (height, width)
            sampling_ratio (int): number of sampling points in the interpolation grid
                used to compute the output value of each pooled output bin. If > 0,
                then exactly sampling_ratio x sampling_ratio grid points are used. If
                <= 0, then an adaptive number of grid points are used (computed as
                ceil(roi_width / pooled_w), and likewise for height). Default: -1
        """
        
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.spatial_scale = None
        
    def setup_scale(self, feature_shape, image_shape):
        if self.spatial_scale is not None:
            return
        
        possible_scales = []
        for s1, s2 in zip(feature_shape, image_shape):
            scale = 2 ** int(math.log2(s1 / s2))
            possible_scales.append(scale)
        assert possible_scales[0] == possible_scales[1]
        self.spatial_scale = possible_scales[0]
        
    def __call__(self, feature, proposal, image_shape):
        """
        Arguments:
            feature (Tensor[N, C, H, W])
            proposal (Tensor[K, 4])
            image_shape (Torch.Size([H, W]))

        Returns:
            output (Tensor[K, C, self.output_size[0], self.output_size[1]])
        
        """
        if isinstance(feature, dict):
            feature = list(feature.values()).pop()

        idx = proposal.new_full((proposal.shape[0], 1), 0)
        roi = torch.cat((idx, proposal), dim=1)
        
        self.setup_scale(feature.shape[-2:], image_shape)
        return roi_align(feature.to(roi), roi, self.spatial_scale, self.output_size[0], self.output_size[1], self.sampling_ratio)


class MultiScaleRoIAlign:

    def __init__(self,
                 featmap_names,
                 output_size,
                 sampling_ratio,
                 *,
                 canonical_scale = 224,
                 canonical_level = 4,
                 ):

        self.featmap_names = featmap_names
        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)
        self.scales = None
        self.map_levels = None
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level
        self.spatial_scale = None

    def infer_scale(self, feature, original_size) -> float:
        # assumption: the scale is of the form 2 ** (-k), with k integer
        size = feature.shape[-2:]
        possible_scales = []
        for s1, s2 in zip(size, original_size):
            approx_scale = float(s1) / float(s2)
            scale = 2 ** float(torch.tensor(approx_scale).log2().round())
            possible_scales.append(scale)
        assert possible_scales[0] == possible_scales[1]
        return possible_scales[0]

    def setup_scales( self, features, image_shapes):

        assert len(image_shapes) != 0
        max_x = image_shapes[0]
        max_y = image_shapes[1]
        original_input_shape = (max_x, max_y)

        scales = [self.infer_scale(feat, original_input_shape) for feat in features]
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.scales = scales
        self.map_levels = initLevelMapper(
            int(lvl_min),
            int(lvl_max),
            canonical_scale=self.canonical_scale,
            canonical_level=self.canonical_level,
        )

    def convert_to_roi_format(self, boxes):
        idx = boxes.new_full((boxes.shape[0], 1), 0)
        rois = torch.cat((idx, boxes), dim=1)
        return rois

    def __call__(self, x, boxes, image_shapes):
        """
        Arguments:
            x (Tensor[N, C, H, W])
            boxes (Tensor[K, 4])
            image_shape (Torch.Size([H, W]))

        Returns:
            output (Tensor[K, C, self.output_size[0], self.output_size[1]])

        """

        x_filtered = []
        for k, v in x.items():
            if k in self.featmap_names:
                x_filtered.append(v)
        num_levels = len(x_filtered)

        rois = self.convert_to_roi_format(boxes)
        num_rois = len(rois)
        num_channels = x_filtered[0].shape[1]

        image_shapes = list(image_shapes)
        if self.scales is None:
            self.setup_scales(x_filtered, image_shapes)

        scales = self.scales
        assert scales is not None

        mapper = self.map_levels
        assert mapper is not None
        boxes = list(boxes)
        levels = mapper(boxes)

        dtype, device = x_filtered[0].dtype, x_filtered[0].device
        result = torch.zeros(
            (num_rois, num_channels,) + self.output_size,
            dtype=dtype,
            device=device,
            )

        # print()
        for level, (per_level_feature, scale) in enumerate(zip(x_filtered, scales)):

            idx_in_level = torch.where(levels == level)[0]
            rois_per_level = rois[idx_in_level]
            # print(f'rois_per_level: {level} {rois_per_level.size()}')

            result_idx_in_level = roi_align(per_level_feature.to(rois_per_level),
                                            rois_per_level, scale,
                                            self.output_size[0],
                                            self.output_size[1],
                                            self.sampling_ratio)

            result[idx_in_level] = result_idx_in_level.to(result.dtype)

        return result


def initLevelMapper(k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
    return LevelMapper(k_min, k_max, canonical_scale, canonical_level, eps)


class LevelMapper():
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    Args:
        k_min (int)
        k_max (int)
        canonical_scale (int)
        canonical_level (int)
        eps (float)
    """

    def __init__(self,k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):

        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists):
        """
        Args:
            boxlists (list[BoxList])
        """
        # Compute level ids
        s = torch.sqrt(torch.cat([box_area(boxlist.view(1, 4)) for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0) + torch.tensor(self.eps, dtype=s.dtype))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return (target_lvls.to(torch.int64) - self.k_min).to(torch.int64)
