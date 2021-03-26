
import yaml
import numpy as np

#######################################
#######################################

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).resolve().parents[1]

###########################################################
# obj bbox
###########################################################

def get_obj_bbox(mask, obj_id, img_width, img_length, border_list):

    ####################
    ## affordance id
    ####################

    rows = np.any(mask==obj_id, axis=1)
    cols = np.any(mask==obj_id, axis=0)

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
    # return [x1, y1, x2, y2]
    boxes = np.zeros([1, 4], dtype=np.int32)
    boxes[0] = np.array([x1, y1, x2, y2])
    return boxes.astype(np.int32)