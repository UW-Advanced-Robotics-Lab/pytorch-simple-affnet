import numpy as np

from utils import helper_utils

#######################################
# OBJECT CONFIGS
#######################################

DRAW_OBJ_PART_POSE = np.array([1, 3, 5, 7, 9, 11, 14, 16, 19, 22, 24])

##################################
##################################

MODIFY_OBJECT_POSE = np.array([6, 7, 8, 9, 10, 11])

def modify_obj_rotation_matrix_for_grasping(obj_id, obj_r):

    # theta = np.pi/2
    # ccw_x_rotation = np.array([[1, 0, 0],
    #                            [0, np.cos(theta), -np.sin(theta)],
    #                            [0, np.sin(theta), np.cos(theta)],
    #                            ])
    #
    # ccw_y_rotation = np.array([[np.cos(theta), 0 , np.sin(theta)],
    #                            [0, 1, 0],
    #                            [-np.sin(theta), 0, np.cos(theta)],
    #                            ])
    #
    # ccw_z_rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
    #                            [np.sin(theta), np.cos(theta), 0],
    #                            [0, 0, 1],
    #                            ])

    if obj_id in np.array([6, 7, 8, 11]): # 019_pitcher_base, 024_bowl, 025_mug or 051_large_clamp
        # rotate about z-axis
        theta = -np.pi / 2
        ccw_z_rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                                   [np.sin(theta), np.cos(theta), 0],
                                   [0, 0, 1],
                                   ])
        obj_r = np.dot(obj_r, ccw_z_rotation)

        # rotate about x-axis
        theta = -np.pi / 2
        ccw_x_rotation = np.array([[1, 0, 0],
                                   [0, np.cos(theta), -np.sin(theta)],
                                   [0, np.sin(theta), np.cos(theta)],
                                   ])
        obj_r = np.dot(obj_r, ccw_x_rotation)

    elif obj_id == 9:  # 035_power_drill
        # rotate about y-axis
        theta = np.pi / 2
        ccw_y_rotation = np.array([[np.cos(theta), 0 , np.sin(theta)],
                                   [0, 1, 0],
                                   [-np.sin(theta), 0, np.cos(theta)],
                                   ])
        obj_r = np.dot(obj_r, ccw_y_rotation)

        # # rotate about x-axis
        theta = np.pi
        ccw_x_rotation = np.array([[1, 0, 0],
                                   [0, np.cos(theta), -np.sin(theta)],
                                   [0, np.sin(theta), np.cos(theta)],
                                   ])
        obj_r = np.dot(obj_r, ccw_x_rotation)

    elif obj_id == 10: # 037_scissors
        # rotate about z-axis
        theta = np.pi / 2
        ccw_z_rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                                   [np.sin(theta), np.cos(theta), 0],
                                   [0, 0, 1],
                                   ])
        obj_r = np.dot(obj_r, ccw_z_rotation)

        # rotate about x-axis
        theta = np.pi / 2
        ccw_x_rotation = np.array([[1, 0, 0],
                                   [0, np.cos(theta), -np.sin(theta)],
                                   [0, np.sin(theta), np.cos(theta)],
                                   ])
        obj_r = np.dot(obj_r, ccw_x_rotation)

        # rotate about y-axis
        theta = -np.pi / 10
        ccw_y_rotation = np.array([[np.cos(theta), 0, np.sin(theta)],
                                   [0, 1, 0],
                                   [-np.sin(theta), 0, np.cos(theta)],
                                   ])
        obj_r = np.dot(obj_r, ccw_y_rotation)

    return obj_r

##################################
##################################

def convert_obj_part_mask_to_obj_mask(obj_part_mask):

    obj_part_mask = np.array(obj_part_mask)
    obj_mask = np.zeros((obj_part_mask.shape[0], obj_part_mask.shape[1]), dtype=np.uint8)

    obj_part_ids = np.unique(obj_part_mask)[1:]
    # obj_part_ids = np.flip(obj_part_ids)
    for obj_part_id in obj_part_ids:
        obj_id = map_obj_part_id_to_obj_id(obj_part_id)
        # print(f'obj_part_id:{obj_part_id}, obj_id:{obj_id}')
        aff_mask_one = np.ones((obj_part_mask.shape[0], obj_part_mask.shape[1]), dtype=np.uint8)
        aff_mask_one = aff_mask_one * obj_id
        obj_mask = np.where(obj_part_mask==obj_part_id, aff_mask_one, obj_mask).astype(np.uint8)
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
    elif object_id == 2:        # 002_spatula
        return 'spatula'
    elif object_id == 3:        # 003_wooden_spoon
        return 'wooden_spoon'
    elif object_id == 4:        # 004_screwdriver
        return 'screwdriver'
    elif object_id == 5:        # 005_garden_shovel
        return 'garden_shovel'
    elif object_id == 6:        # 019_pitcher_base
        return 'pitcher'
    elif object_id == 7:        # 024_bowl
        return 'bowl'
    elif object_id == 8:        # 025_mug
        return 'mug'
    elif object_id == 9:        # 035_power_drill
        return 'power_drill'
    elif object_id == 10:       # 037_scissors
        return 'scissors'
    elif object_id == 11:       # 051_large_clamp
        return 'large_clamp'
    else:
        print(f" --- Object ID:{object_id} does not map to Object Label --- ")
        exit(1)

def map_aff_id_to_name(aff_id):

    if aff_id == 1:
        return 'grasp'
    elif aff_id == 2:
        return 'screw'
    elif aff_id == 3:
        return 'scoop'
    elif aff_id == 4:
        return 'pound'
    elif aff_id == 5:
        return 'support'
    elif aff_id == 6:
        return 'cut'
    elif aff_id == 7:
        return 'wrap-grasp'
    elif aff_id == 8:
        return 'contain'
    elif aff_id == 9:
        return 'clamp'
    else:
        print(f" --- Affordance ID:{aff_id} does not map to Object Label --- ")
        exit(1)

##################################
##################################

def format_obj_ids_to_aff_ids_list(object_ids, gt_object_part_ids):
    if len(object_ids) == 0:
        return []
    else:
        _current_idx = 0
        _aff_ids_list = []
        _obj_part_ids_list = []
        for i, object_id in enumerate(object_ids):
            object_part_ids = map_obj_id_to_obj_part_ids(object_id)
            # _aff_ids_list.append(list(aff_ids[_current_idx:(_current_idx+len(object_part_ids))]))
            # _current_idx += len(object_part_ids)
            _obj_part_temp, _aff_temp = [], []
            for object_part_id in object_part_ids:
                if object_part_id in gt_object_part_ids:
                    _obj_part_temp.append(object_part_id)
                    _aff_temp.append(map_obj_part_id_to_aff_id(object_part_id))
            _obj_part_ids_list.append(_obj_part_temp)
            _aff_ids_list.append(_aff_temp)
        return _obj_part_ids_list, _aff_ids_list

def map_obj_ids_to_aff_ids_list(object_ids):
    if len(object_ids) == 0:
        return []
    else:
        _aff_ids_list = []
        _obj_part_ids_list = []
        for object_id in object_ids:
            _obj_part_temp = []
            object_part_ids = map_obj_id_to_obj_part_ids(object_id)
            for object_part_id in object_part_ids:
                _obj_part_temp.append(map_obj_part_id_to_aff_id(object_part_id))
                # _object_part_ids_list.append(object_part_id)
            _obj_part_ids_list.append(object_part_ids)
            _aff_ids_list.append(_obj_part_temp)
        return _obj_part_ids_list, _aff_ids_list

def map_obj_id_to_obj_part_ids(object_id):

    if object_id == 1:          # 001_mallet
        return [1, 2]
    elif object_id == 2:        # 002_spatula
        return [3, 4]
    elif object_id == 3:        # 003_wooden_spoon
        return [5, 6]
    elif object_id == 4:        # 004_screwdriver
        return [7, 8]
    elif object_id == 5:        # 005_garden_shovel
        return [9, 10]
    elif object_id == 6:        # 019_pitcher_base
        # return [11, 12, 13]
        return [12, 11, 13]
    elif object_id == 7:        # 024_bowl
        return [14, 15]
    elif object_id == 8:        # 025_mug
        # return [16, 17, 18]
       return [17, 16, 18]
    elif object_id == 9:        # 035_power_drill
        # return [19, 20, 21]
        return [21, 19, 20]
    elif object_id == 10:       # 037_scissors
        return [22, 23]
    elif object_id == 11:       # 051_large_clamp
        return [24, 25]
    else:
        print(" --- Object ID does not map to Object Part IDs --- ")
        exit(1)

def map_obj_part_id_to_obj_id(obj_part_id):

    if obj_part_id == 0:  # 001_mallet
        return 0
    elif obj_part_id in [1, 2]:          # 001_mallet
        return 1
    elif obj_part_id in [3, 4]:        # 002_spatula
        return 2
    elif obj_part_id in [5, 6]:        # 003_wooden_spoon
        return 3
    elif obj_part_id in [7, 8]:        # 004_screwdriver
        return 4
    elif obj_part_id in [9, 10]:       # 005_garden_shovel
        return 5
    elif obj_part_id in [12, 11, 13]: # 019_pitcher_base
        return 6
    elif obj_part_id in [14, 15]:     # 024_bowl
        return 7
    elif obj_part_id in [17, 16, 18]: # 025_mug
        return 8
    elif obj_part_id in [21, 19, 20]: # 035_power_drill
        return 9
    elif obj_part_id in [22, 23]:     # 037_scissors
        return 10
    elif obj_part_id in [24, 25]:     # 051_large_clamp
        return 11
    else:
        print(" --- Object Part ID does not map to Object ID --- ")
        exit(1)

def map_obj_part_id_to_aff_id(obj_part_id):

    if obj_part_id in [1, 3, 5, 7, 9, 11, 16, 19, 22]:  # grasp
        return 1
    elif obj_part_id in [8, 20]:                        # screw
        return 2
    elif obj_part_id in [6, 10]:                        # scoop
        return 3
    elif obj_part_id in [2]:                            # pound
        return 4
    elif obj_part_id in [4, 21]:                        # support
        return 5
    elif obj_part_id in [23]:                           # cut
        return 6
    elif obj_part_id in [12, 14, 17, 24]:               # wrap-grasp
        return 7
    elif obj_part_id in [13, 15, 18]:                   # contain
        return 8
    elif obj_part_id in [25]:                           # clamp
        return 9
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

    return np.squeeze(color_mask)

def obj_color_map_dict():
    ''' [red, blue, green]'''

    obj_color_map_dict = {
        0: [0, 0, 0],
        1: [235, 34, 17],   # red
        2: [235, 96, 17],   # orange
        3: [235, 195, 17],  # gold
        4: [176, 235, 17],  # light green/yellow
        5: [76, 235, 17],   # green
        6: [17, 235, 139],  # teal
        7: [17, 235, 225],  # light blue
        8: [17, 103, 235],  # dark blue
        9: [133, 17, 235],  # purple
        10: [235, 17, 215],  # pink
        11: [235, 17, 106],  # hot pink
    }

    return obj_color_map_dict

##################################
##################################

def obj_color_map(idx):
    # print(f'idx:{idx}')
    ''' [red, blue, green]'''

    if idx == 1:
        return (235, 34, 17)        # red
    elif idx == 2:
        return (235, 96, 17)        # orange
    elif idx == 3:
        return (235, 195, 17)       # gold
    elif idx == 4:
        return (176,  235, 17)      # light green/yellow
    elif idx == 5:
        return (76,   235, 17)      # green
    elif idx == 6:
        return (17,  235, 139)      # teal
    elif idx == 7:
        return (17,  235, 225)      # light blue
    elif idx == 8:
        return (17,  103, 235)      # dark blue
    elif idx == 9:
        return (133,  17, 235)      # purple
    elif idx == 10:
        return (235, 17, 215)       # pink
    elif idx == 11:
        return (235, 17, 106)       # hot pink
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
        1: [133, 17, 235],   # red
        2: [235, 96, 17],   # orange
        3: [235, 195, 17],  # gold
        4: [176, 235, 17],  # light green/yellow
        5: [76, 235, 17],   # green
        6: [17, 235, 139],  # teal
        7: [17, 235, 225],  # light blue
        8: [17, 103, 235],  # dark blue
        9: [235, 34, 17],  # purple
    }

    return aff_color_map_dict

##################################
##################################

def aff_color_map(idx):
    ''' [red, blue, green]'''

    if idx == 1:
        return (133,  17, 235)        # red
    elif idx == 2:
        return (235, 96, 17)        # orange
    elif idx == 3:
        return (235, 195, 17)       # gold
    elif idx == 4:
        return (176,  235, 17)      # light green/yellow
    elif idx == 5:
        return (76,   235, 17)      # green
    elif idx == 6:
        return (17,  235, 139)      # teal
    elif idx == 7:
        return (17,  235, 225)      # light blue
    elif idx == 8:
        return (17,  103, 235)      # dark blue
    elif idx == 9:
        return (235, 34, 17)      # purple
    else:
        print(f" --- Affordance ID:{idx} does not map to a colour --- ")
        exit(1)