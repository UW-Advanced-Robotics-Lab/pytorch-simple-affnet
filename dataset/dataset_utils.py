import numpy as np

from PIL import Image

from imgaug import augmenters as iaa

import torchvision
from torchvision.transforms import functional as F


def convert_16_bit_depth_to_8_bit(depth):
    depth = np.array(depth, np.uint16)
    depth = depth / np.max(depth) * (2 ** 8 - 1)
    return np.array(depth, np.uint8)

def print_depth_info(depth):
    depth = np.array(depth)
    print(f"Depth of type:{depth.dtype} has min:{np.min(depth)} & max:{np.max(depth)}")

def print_class_labels(seg_mask):
    class_ids = np.unique(np.array(seg_mask, dtype=np.uint8))[1:]  # exclude the background
    print(f"Mask has {len(class_ids)} Labels: {class_ids}")

def format_output_maskrcnn_data(image, output):
    pass

def format_output_affnet_data(image, output):
    pass

def format_label(label):
    return np.array(label, dtype=np.int32)

def format_bbox(bbox):
    return np.array(bbox, dtype=np.int32).flatten()

def crop(pil_img, crop_size, is_img=False):
    _dtype = np.array(pil_img).dtype
    pil_img = Image.fromarray(pil_img)
    crop_w, crop_h = crop_size
    img_width, img_height = pil_img.size
    left, right = (img_width - crop_w) / 2, (img_width + crop_w) / 2
    top, bottom = (img_height - crop_h) / 2, (img_height + crop_h) / 2
    left, top = round(max(0, left)), round(max(0, top))
    right, bottom = round(min(img_width - 0, right)), round(min(img_height - 0, bottom))
    # pil_img = pil_img.crop((left, top, right, bottom)).resize((crop_w, crop_h))
    pil_img = pil_img.crop((left, top, right, bottom))
    ###
    if is_img:
        img_channels = np.array(pil_img).shape[-1]
        img_channels = 3 if img_channels == 4 else img_channels
        resize_img = np.zeros((crop_h, crop_w, img_channels))
        resize_img[0:(bottom - top), 0:(right - left), :img_channels] = np.array(pil_img)[..., :img_channels]
    else:
        resize_img = np.zeros((crop_h, crop_w))
        resize_img[0:(bottom - top), 0:(right - left)] = np.array(pil_img)

    return np.array(resize_img, dtype=_dtype)

def get_image_augmentations():
    # imgaug to labels.
    affine = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.Flipud(0.25),  # vertical flip 50% of the images
        iaa.Crop(percent=(0.00, 0.10)),  # random crops
        # iaa.Affine(
        #     # scale={"x": (0.75, 1.25), "y": (0.75, 1.25)},
        #     # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        #     rotate=(-5, 5),
        #     # shear=(-3, 3)
        # )
    ], random_order=True)

    # imgaug to rgb images.
    colour_aug = iaa.Sequential([
        # Small gaussian blur with random sigma between 0 and 3.0.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 3))
        ),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Strengthen or weaken the contrast in each image.
        # iaa.LinearContrast((0.75, 1.5)),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        # iaa.Multiply((0.8, 1.2), per_channel=0.2),
    ], random_order=True)  # apply augmenters in random order

    # imgaug to depth images.
    depth_aug = iaa.Sometimes(0.833, iaa.Sequential([
        iaa.CoarseDropout(5e-4, size_percent=0.5),
        iaa.SaltAndPepper(5e-4),
    ], random_order=True))  # apply augmenters in random order

    return affine, colour_aug, depth_aug

def get_bbox(mask, obj_ids, img_width, img_height, _step=40):
    border_list = np.arange(start=0, stop=np.max([img_width, img_height]) + _step, step=_step)
    # init boxes.
    boxes = np.zeros([len(obj_ids), 4], dtype=np.int32)
    for idx, obj_id in enumerate(obj_ids):
        rows = np.any(mask == obj_id, axis=1)
        cols = np.any(mask == obj_id, axis=0)

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        y2 += 1
        x2 += 1
        r_b = y2 - y1
        for tt in range(len(border_list)):
            if r_b > border_list[tt] and r_b < border_list[tt + 1]:
                r_b = border_list[tt + 1]
                break
        c_b = x2 - x1
        for tt in range(len(border_list)):
            if c_b > border_list[tt] and c_b < border_list[tt + 1]:
                c_b = border_list[tt + 1]
                break
        center = [int((y1 + y2) / 2), int((x1 + x2) / 2)]
        y1 = center[0] - int(r_b / 2)
        y2 = center[0] + int(r_b / 2)
        x1 = center[1] - int(c_b / 2)
        x2 = center[1] + int(c_b / 2)
        if y1 < 0:
            delt = -y1
            y1 = 0
            y2 += delt
        if x1 < 0:
            delt = -x1
            x1 = 0
            x2 += delt
        if y2 > img_width:
            delt = y2 - img_width
            y2 = img_width
            y1 -= delt
        if x2 > img_height:
            delt = x2 - img_height
            x2 = img_height
            x1 -= delt
        # x1,y1 ------
        # |          |
        # |          |
        # |          |
        # --------x2,y2
        # cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (255, 0, 0), 2)
        boxes[idx] = np.array([x1, y1, x2, y2])
    return boxes

def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform():
    transforms = []
    transforms.append(ToTensor())
    return Compose(transforms)

class ToTensor(object):
        def __call__(self, image, target):
            image = F.to_tensor(image)
            return image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target




