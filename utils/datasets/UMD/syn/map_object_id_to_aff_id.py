import numpy as np
import shutil
import glob
import os

import skimage.io

import matplotlib.pyplot as plt
import cv2

debug = False

###########################################################
# FOR SYTHENTIC IMAGES
# LOOKUP FROM OBJECT ID TO AFFORDANCE LABEL
###########################################################

def seq_get_masks(og_mask):

    instance_masks = np.zeros((og_mask.shape[0], og_mask.shape[1]), dtype=np.uint8)
    instance_mask_one = np.ones((og_mask.shape[0], og_mask.shape[1]), dtype=np.uint8)

    object_id_labels = np.unique(og_mask)
    # print("GT Object ID:", np.unique(object_id_labels))

    for i, object_id in enumerate(object_id_labels):
        if object_id != 0:
            affordance_id = map_affordance_label(object_id)
            # print("Affordance Label:", affordance_id)

            instance_mask = instance_mask_one * affordance_id
            instance_masks = np.where(og_mask==object_id, instance_mask, instance_masks).astype(np.uint8)

            # idx = np.where(og_mask == object_id)[0]
            # instance_masks[idx] = affordance_id

    return instance_masks.astype(np.uint8)

def map_affordance_label(current_id):

    # 1
    grasp = [
        20,  # "hammer-grasp"
        22,  # "hammer-grasp"
        24,  # "hammer-grasp"
        26,  # "hammer-grasp"
        #
        28,  # "knife-grasp"
        30,  # "knife-grasp"
        32,  # "knife-grasp"
        34,  # "knife-grasp"
        36,  # "knife-grasp"
        38,  # "knife-grasp"
        40,  # "knife-grasp"
        42,  # "knife-grasp"
        44,  # "knife-grasp"
        46,  # "knife-grasp"
        48,  # "knife-grasp"
        50,  # "knife-grasp"
        #
        52,  # "ladle-grasp"
        54,  # "ladle-grasp"
        56,  # "ladle-grasp"
        #
        58,  # "mallet-grasp"
        60,  # "mallet-grasp"
        62,  # "mallet-grasp"
        64,  # "mallet-grasp"
        #
        66,  # "mug-grasp"
        69,  # "mug-grasp"
        72,  # "mug-grasp"
        75,  # "mug-grasp"
        78,  # "mug-grasp"
        81,  # "mug-grasp"
        84,  # "mug-grasp"
        87,  # "mug-grasp"
        90,  # "mug-grasp"
        93,  # "mug-grasp"
        96,  # "mug-grasp"
        99,  # "mug-grasp"
        102,  # "mug-grasp"
        105,  # "mug-grasp"
        108,  # "mug-grasp"
        111,  # "mug-grasp"
        114,  # "mug-grasp"
        117,  # "mug-grasp"
        120,  # "mug-grasp"
        123,  # "mug-grasp"
        #
        130,  # "saw-grasp"
        132,  # "saw-grasp"
        134,  # "saw-grasp"
        #
        136,  # "scissors-grasp"
        138,  # "scissors-grasp"
        140,  # "scissors-grasp"
        142,  # "scissors-grasp"
        144,  # "scissors-grasp"
        146,  # "scissors-grasp"
        148,  # "scissors-grasp"
        150,  # "scissors-grasp"
        #
        152,  # "scoop-grasp"
        154,  # "scoop-grasp"
        #
        156,  # "shears-grasp"
        158,  # "shears-grasp"
        #
        160,  # "shovel-grasp"
        162,  # "shovel-grasp"
        #
        164,  # "spoon-grasp"
        166,  # "spoon-grasp"
        168,  # "spoon-grasp"
        170,  # "spoon-grasp"
        172,  # "spoon-grasp"
        174,  # "spoon-grasp"
        176,  # "spoon-grasp"
        178,  # "spoon-grasp"
        180,  # "spoon-grasp"
        182,  # "spoon-grasp"
        #
        184,  # "tenderizer-grasp"
        #
        186,  # "trowel-grasp"
        188,  # "trowel-grasp"
        190,  # "trowel-grasp"
        #
        192,  # "turner-grasp"
        194,  # "turner-grasp"
        196,  # "turner-grasp"
        198,  # "turner-grasp"
        200,  # "turner-grasp"
        202,  # "turner-grasp"
        204,  # "turner-grasp"
    ]

    # 2
    cut = [
        28 + 1,  # "knife-cut"
        30 + 1,  # "knife-cut"
        32 + 1,  # "knife-cut"
        34 + 1,  # "knife-cut"
        36 + 1,  # "knife-cut"
        38 + 1,  # "knife-cut"
        40 + 1,  # "knife-cut"
        42 + 1,  # "knife-cut"
        44 + 1,  # "knife-cut"
        46 + 1,  # "knife-cut"
        48 + 1,  # "knife-cut"
        50 + 1,  # "knife-cut"
        #
        130 + 1,  # "saw-cut"
        132 + 1,  # "saw-cut"
        134 + 1,  # "saw-cut"
        #
        136 + 1,  # "scissors-cut"
        138 + 1,  # "scissors-cut"
        140 + 1,  # "scissors-cut"
        142 + 1,  # "scissors-cut"
        144 + 1,  # "scissors-cut"
        146 + 1,  # "scissors-cut"
        148 + 1,  # "scissors-cut"
        150 + 1,  # "scissors-cut"
        #
        156 + 1, # "shears-cut"
        158 + 1,
    ]

    # 3
    scoop = [
        152 + 1,  # "scoop-scoop"
        154 + 1,  # "scoop-scoop"
        #
        160 + 1,  # "shovel-scoop"
        162 + 1,  # "shovel-scoop"
        #
        164 + 1,  # "spoon-scoop"
        166 + 1,  # "spoon-scoop"
        168 + 1,  # "spoon-scoop"
        170 + 1,  # "spoon-scoop"
        172 + 1,  # "spoon-scoop"
        174 + 1,  # "spoon-scoop"
        176 + 1,  # "spoon-scoop"
        178 + 1,  # "spoon-scoop"
        180 + 1,  # "spoon-scoop"
        182 + 1,  # "spoon-scoop"
        #
        186 + 1,  # "trowel-scoop"
        188 + 1,  # "trowel-scoop"
        190 + 1,  # "trowel-scoop"
    ]

    # 4
    contain = [
        1,  # "bowl-contain"
        2,  # "bowl-contain"
        3,  # "bowl-contain"
        4,  # "bowl-contain"
        5,  # "bowl-contain"
        6,  # "bowl-contain"
        7,  # "bowl-contain"
        #
        8,  # "cup-contain"
        10,  # "cup-contain"
        12,  # "cup-contain"
        14,  # "cup-contain"
        16,  # "cup-contain"
        18,  # "cup-contain"
        #
        52 + 1,  # "ladle-contain"
        54 + 1,  # "ladle-contain"
        56 + 1,  # "ladle-contain"
        66 + 1,  # "mug-contain"
        69 + 1,  # "mug-contain"
        72 + 1,  # "mug-contain"
        75 + 1,  # "mug-contain"
        78 + 1,  # "mug-contain"
        81 + 1,  # "mug-contain"
        84 + 1,  # "mug-contain"
        87 + 1,  # "mug-contain"
        90 + 1,  # "mug-contain"
        93 + 1,  # "mug-contain"
        96 + 1,  # "mug-contain"
        99 + 1,  # "mug-contain"
        #
        102 + 1,  # "mug-contain"
        105 + 1,  # "mug-contain"
        108 + 1,  # "mug-contain"
        111 + 1,  # "mug-contain"
        114 + 1,  # "mug-contain"
        117 + 1,  # "mug-contain"
        120 + 1,  # "mug-contain"
        123 + 1,  # "mug-contain"
        #
        126,  # "pot-contain"
        128,  # "pot-contain"
    ]

    # 5
    pound = [
        20 + 1, #"hammer-pound"
        22 + 1, #"hammer-pound"
        24 + 1, #"hammer-pound"
        26 + 1, #"hammer-pound"
        #
        58 + 1, #'mallet-pound'
        60 + 1, #'mallet-pound'
        62 + 1, #'mallet-pound'
        64 + 1, #'mallet-pound'
        #
        184 + 1, #'tenderizer-pound'
    ]

    # 6
    support = [
        192 + 1,  # "turner-support"
        194 + 1,  # "turner-support"
        196 + 1,  # "turner-support"
        198 + 1,  # "turner-support"
        200 + 1,  # "turner-support"
        202 + 1,  # "turner-support"
        204 + 1,  # "turner-support"
    ]

    # 7
    wrap_grasp = [
        8 + 1, # "cup-wrap_grasp"
        10 + 1, # "cup-wrap_grasp"
        12 + 1, # "cup-wrap_grasp"
        14 + 1, # "cup-wrap_grasp"
        16 + 1, # "cup-wrap_grasp"
        18 + 1, # "cup-wrap_grasp"
        #
        66 + 2, # "mug-wrap_grasp"
        69 + 2, # "mug-wrap_grasp"
        72 + 2, # "mug-wrap_grasp"
        75 + 2, # "mug-wrap_grasp"
        78 + 2, # "mug-wrap_grasp"
        81 + 2, # "mug-wrap_grasp"
        84 + 2, # "mug-wrap_grasp"
        87 + 2, # "mug-wrap_grasp"
        90 + 2, # "mug-wrap_grasp"
        93 + 2, # "mug-wrap_grasp"
        96 + 2, # "mug-wrap_grasp"
        99 + 2, # "mug-wrap_grasp"
        102 + 2, # "mug-wrap_grasp"
        105 + 2, # "mug-wrap_grasp"
        108 + 2, # "mug-wrap_grasp"
        111 + 2, # "mug-wrap_grasp"
        114 + 2, # "mug-wrap_grasp"
        117 + 2, # "mug-wrap_grasp"
        120 + 2, # "mug-wrap_grasp"
        123 + 2, # "mug-wrap_grasp"
        #
        126 + 1, # "pot-wrap_grasp"
        128 + 1, # "pot-wrap_grasp"
    ]

    if current_id in grasp:
        return 1
    elif current_id in cut:
        return 2
    elif current_id in scoop:
        return 3
    elif current_id in contain:
        return 4
    elif current_id in pound:
        return 5
    elif current_id in support:
        return 6
    elif current_id in wrap_grasp:
        return 7
    else:
        print(" --- Object ID does not map to Affordance Label --- ")
        print(current_id)
        exit(1)


###########################################################
#
###########################################################
if __name__ == '__main__':

    ########################
    ########################
    data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds4/umd_affordance/tools/'

    objects = [
        'bowl_01/',
    ]

    scenes = [
        'bench/',
        'floor/',
        'turn_table/',
        'dr/',
    ]

    splits = [
        'train/',
        'val/',
    ]

    cameras = [
        'Kinect/',
        'Xtion/',
        'ZED/',
    ]

    image_exts = [
        '.cs.png',
    ]

    ########################
    ########################

    for scene in scenes:
        for image_ext in image_exts:
            # file_path = data_path + object + scene + split + camera + '??????' + image_ext
            file_path = data_path + '*/' + scene + '*/' + '*/' + '??????' + image_ext
            files = np.array(sorted(glob.glob(file_path)))
            print("\nLoaded files: ", len(files))
            print("File path: ", file_path)

            for idx, file in enumerate(files):
                old_file_name = file
                new_file_name = old_file_name.split(image_ext)[0] + '.aff.png'
                if idx == 0:
                    print(f'old_file_name:{old_file_name}')
                    print(f'new_file_name:{new_file_name}')

                object_id_label = np.array(skimage.io.imread(file))
                affordance_label = seq_get_masks(object_id_label)

                skimage.io.imsave(new_file_name, affordance_label)

                if debug:
                    # print(f'old_file_name:{old_file_name}')
                    # print(f'new_file_name:{new_file_name}')
                    # print("object_id_label: ", np.unique(object_id_label))
                    # print("affordance_label: ", np.unique(affordance_label))
                    cv2.imshow('object', object_id_label*50)
                    cv2.imshow('aff', affordance_label*50)
                    cv2.waitKey(0)
