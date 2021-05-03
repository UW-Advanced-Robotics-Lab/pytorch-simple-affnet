import numpy as np

#######################################
# OBJECT CONFIGS
#######################################

PROJECT_POINT_CLOUD = np.array([1, 2])
DRAW_OBJ_PART_POSE = np.array([1])

##################################
##################################

def convert_obj_part_mask_to_obj_mask(obj_part_mask):

    obj_part_mask = np.array(obj_part_mask)
    obj_mask = np.zeros((obj_part_mask.shape[0], obj_part_mask.shape[1]), dtype=np.uint8)

    obj_part_ids = np.unique(obj_part_mask)[1:]
    for obj_part_id in obj_part_ids:
        obj_id = map_obj_part_id_to_obj_id(obj_part_id)
        # print(f'obj_part_id:{obj_part_id}, obj_id:{obj_id}')
        obj_mask_one = np.ones((obj_part_mask.shape[0], obj_part_mask.shape[1]), dtype=np.uint8)
        obj_mask_one = obj_mask_one * obj_id
        obj_mask = np.where(obj_part_mask==obj_part_id, obj_mask_one, obj_mask).astype(np.uint8)
    # helper_utils.print_class_labels(obj_mask)
    return obj_mask

def convert_obj_part_mask_to_aff_mask(obj_part_mask):

    obj_part_mask = np.array(obj_part_mask)
    aff_mask = np.zeros((obj_part_mask.shape[0], obj_part_mask.shape[1]), dtype=np.uint8)

    obj_part_ids = np.unique(obj_part_mask)[1:]
    # obj_part_ids = np.flip(obj_part_ids)
    for obj_part_id in obj_part_ids:
        aff_id = map_obj_part_id_to_aff_id(obj_part_id)
        # print(f'obj_part_id:{obj_part_id}, obj_id:{aff_id}')
        aff_mask_one = np.ones((obj_part_mask.shape[0], obj_part_mask.shape[1]), dtype=np.uint8)
        aff_mask_one = aff_mask_one * aff_id
        aff_mask = np.where(obj_part_mask==obj_part_id, aff_mask_one, aff_mask).astype(np.uint8)
    # helper_utils.print_class_labels(aff_mask)
    return aff_mask

##################################
##################################

def map_obj_id_to_name(object_id):

    if object_id == 1:          # 001_mallet
        return 'mallet'
    else:
        print(" --- Object ID does not map to Object Label --- ")
        exit(1)

##################################
##################################

def map_obj_id_to_obj_part_ids(object_id):

    if object_id == 1:          # 001_mallet
        return [1, 2]
    else:
        print(" --- Object ID does not map to Object Part IDs --- ")
        exit(1)

def map_obj_part_id_to_obj_id(obj_part_id):

    if obj_part_id in [1, 2]:          # 001_mallet
        return 1
    else:
        print(" --- Object Part ID does not map to Object ID --- ")
        exit(1)

def map_obj_part_id_to_aff_id(obj_part_id):

    if obj_part_id in [1]:   # grasp
        return 1
    elif obj_part_id in [2]: # pound
        return 2
    else:
        print(" --- Object Part ID does not map to Affordance ID --- ")
        exit(1)

##################################
##################################

def colorize_obj_mask(instance_mask):

    instance_to_color = obj_color_map_dict()
    color_mask = np.zeros((instance_mask.shape[0], instance_mask.shape[1], 3), dtype=np.uint8)
    for key in instance_to_color.keys():
        color_mask[instance_mask == key] = instance_to_color[key]

    return np.array(np.squeeze(color_mask), dtype=np.uint8)

def obj_color_map_dict():
    ''' [red, blue, green]'''

    obj_color_map_dict = {
        0: [0, 0, 0],
        1: [235, 96, 17],   # orange
    }

    return obj_color_map_dict

##################################
##################################

def obj_color_map(idx):
    # print(f'idx:{idx}')
    ''' [red, blue, green]'''

    if idx == 1:
        return (235, 96, 17)        # orange
    else:
        print(f" --- Object ID:{idx} does not map to a colour --- ")
        exit(1)

##################################
##################################

def colorize_aff_mask(instance_mask):

    instance_to_color = aff_color_map_dict()
    color_mask = np.zeros((instance_mask.shape[0], instance_mask.shape[1], 3), dtype=np.uint8)
    for key in instance_to_color.keys():
        color_mask[instance_mask == key] = instance_to_color[key]

    return np.squeeze(color_mask)

def aff_color_map_dict():
    ''' [red, blue, green]'''

    aff_color_map_dict = {
        0: [0, 0, 0],
        1: [51, 255, 172],  # light green
        2: [141, 51, 255],  # purple
    }

    return aff_color_map_dict

##################################
##################################

def aff_color_map(idx):
    ''' [red, blue, green]'''

    if idx == 1:
        return (51, 255, 172)        # light green
    elif idx == 2:
        return (141, 51, 255)        # purple
    else:
        print(f" --- Affordance ID:{idx} does not map to a colour --- ")
        exit(1)