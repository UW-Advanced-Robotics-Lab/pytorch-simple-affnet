import numpy as np

###########################################################
# bbox
###########################################################

def extract_object_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([1, 4], dtype=np.int32)
    # for i in range(mask.shape[0]):
    m = np.array(mask, dtype=np.uint8)
    # Bounding box.
    horizontal_indicies = np.where(np.any(m, axis=0))[0]
    vertical_indicies = np.where(np.any(m, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    # new_box format: (xmin, ymin, xmax, ymax)
    boxes[0] = np.array([x1, y1, x2, y2])
    return boxes.astype(np.int32)


def extract_aff_bboxes(image, mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    step = 40
    width, height = image.shape[:2]
    border_list = np.arange(start=0, stop=np.max([width, height])+step, step=step)
    aff_labels = np.unique(mask)[1:]
    boxes = np.zeros([len(aff_labels), 4], dtype=np.int32)
    for i, aff_label in enumerate(aff_labels):
        x1, y1, x2, y2 = get_bbox(mask, aff_label, width, height, border_list)
        # new_box format: (xmin, ymin, xmax, ymax)
        boxes[i] = np.array([x1, y1, x2, y2])
    return boxes.astype(np.int32)

###########################################################
###########################################################

def get_bbox(mask, affordance_id, img_width, img_length, border_list):

    ####################
    ## affordance id
    ####################

    rows = np.any(mask==affordance_id, axis=1)
    cols = np.any(mask==affordance_id, axis=0)

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
    if x2 > img_length:
        delt = x2 - img_length
        x2 = img_length
        x1 -= delt
    # x1,y1 ------
    # |          |
    # |          |
    # |          |
    # --------x2,y2
    # cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return x1, y1, x2, y2