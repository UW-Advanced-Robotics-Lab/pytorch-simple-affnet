import numpy as np

##################################
##################################

def object_id_to_name(object_id):

    if object_id == 1:
        return 'Elevator_down_button'
    else:
        print(" --- Object ID does not map to Object Label --- ")
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
    }

    return obj_color_map_dict

##################################
##################################

def obj_color_map(idx):
    ''' [red, blue, green]'''

    if idx == 1:
        return (235, 34, 17)        # red
    else:
        print(" --- idx does not map to a colour --- ")
        exit(1)