import numpy as np
import shutil
import glob
import os

import scipy.io
import scipy.misc
from PIL import Image
import cv2

import matplotlib.pyplot as plt

########################
########################
data_path = '/data/Akeaveny/Datasets/UMD/unformatted/Syn/umd_affordance/tools/*/'
new_data_path = '/data/Akeaveny/Datasets/UMD/Syn/'

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
]

cameras = [
    'Kinect/',
    'Xtion/',
    'ZED/',
]

image_exts = [
    '.png',
    '.depth.png',
    '.aff.png',
    # '.cs.png',
]

train_val_split = 0.7
val_test_split = 0.85 # 30% val split are val / test

########################
########################
offset_train, offset_val, offset_test = 0, 0, 0
train_files_len, val_files_len, test_files_len = 0, 0, 0

for scene in scenes:
    for image_ext in image_exts:
        # file_path = data_path + object + scene + split + camera + '??????' + image_ext
        file_path = data_path   + '*/'   + scene + '*/'  + '*/'   + '??????' + image_ext
        files = np.array(sorted(glob.glob(file_path)))
        print("\nLoaded files: ", len(files))
        print("File path: ", file_path)

        ###############
        # split files
        ###############
        np.random.seed(0)
        total_idx = np.arange(0, len(files), 1)
        train_idx = np.random.choice(total_idx, size=int(train_val_split * len(total_idx)), replace=False)
        val_test_idx = np.delete(total_idx, train_idx)

        train_files = files[train_idx]
        val_test_files = files[val_test_idx]

        val_test_idx = np.arange(0, len(val_test_files), 1)
        val_idx = np.random.choice(val_test_idx, size=int(val_test_split * len(val_test_idx)), replace=False)
        test_idx = np.delete(val_test_idx, val_idx)
        val_files = val_test_files[val_idx]
        test_files = val_test_files[test_idx]

        print("Offset: {}, Chosen Train Files {}/{}".format(offset_train, len(train_files), len(files)))
        print("Offset: {}, Chosen Val Files {}/{}".format(offset_val, len(val_files), len(files)))
        print("Offset: {}, Chosen Test Files {}/{}".format(offset_test, len(test_files), len(files)))

        if image_ext == '.aff.png':
            train_files_len = len(train_files)
            val_files_len = len(val_files)
            test_files_len = len(test_files)

        ###############
        # train
        ###############
        split_folder = 'train/'

        for idx, file in enumerate(train_files):
            old_file_name = file
            new_file_name = new_data_path + split_folder

            object = old_file_name.split('/')[-5].split('_')[0]

            count = 1000000 + offset_train + idx
            image_num = str(count)[1:]

            if image_ext == '.png':
                move_file_name = new_file_name + 'rgb/' + np.str(object) + '_' + np.str(image_num) + '.png'
                if idx == 0:
                    print(f'Old file: {old_file_name}')
                    print(f'New file: {move_file_name}')
                shutil.copyfile(old_file_name, move_file_name)

            elif image_ext == '.depth.png':
                move_file_name = new_file_name + 'depth/' + np.str(object) + '_' + np.str(image_num) + '_depth.png'
                if idx == 0:
                    print(f'Old file: {old_file_name}')
                    print(f'New file: {move_file_name}')
                shutil.copyfile(old_file_name, move_file_name)

            elif image_ext == '.aff.png':
                move_file_name = new_file_name + 'masks/' + np.str(object) + '_' + np.str(image_num) + '_label.png'
                if idx == 0:
                    print(f'Old file: {old_file_name}')
                    print(f'New file: {move_file_name}')
                shutil.copyfile(old_file_name, move_file_name)

            else:
                assert "*** IMAGE EXT DOESN'T EXIST ***"

        ###############
        # val
        ###############
        split_folder = 'val/'

        for idx, file in enumerate(val_files):
            old_file_name = file
            new_file_name = new_data_path + split_folder

            object = old_file_name.split('/')[-5].split('_')[0]

            count = 1000000 + offset_val + idx
            image_num = str(count)[1:]

            if image_ext == '.png':
                move_file_name = new_file_name + 'rgb/' + np.str(object) + '_' + np.str(image_num) + '.png'
                if idx == 0:
                    print(f'Old file: {old_file_name}')
                    print(f'New file: {move_file_name}')
                shutil.copyfile(old_file_name, move_file_name)

            elif image_ext == '.depth.png':
                move_file_name = new_file_name + 'depth/' + np.str(object) + '_' + np.str(
                    image_num) + '_depth.png'
                if idx == 0:
                    print(f'Old file: {old_file_name}')
                    print(f'New file: {move_file_name}')
                shutil.copyfile(old_file_name, move_file_name)

            elif image_ext == '.aff.png':
                move_file_name = new_file_name + 'masks/' + np.str(object) + '_' + np.str(
                    image_num) + '_label.png'
                if idx == 0:
                    print(f'Old file: {old_file_name}')
                    print(f'New file: {move_file_name}')
                shutil.copyfile(old_file_name, move_file_name)

            else:
                assert "*** IMAGE EXT DOESN'T EXIST ***"

        ###############
        # test
        ###############
        split_folder = 'test/'

        for idx, file in enumerate(test_files):
            old_file_name = file
            new_file_name = new_data_path + split_folder

            object = old_file_name.split('/')[-5].split('_')[0]

            count = 1000000 + offset_test + idx
            image_num = str(count)[1:]

            if image_ext == '.png':
                move_file_name = new_file_name + 'rgb/' + np.str(object) + '_' + np.str(image_num) + '.png'
                if idx == 0:
                    print(f'Old file: {old_file_name}')
                    print(f'New file: {move_file_name}')
                shutil.copyfile(old_file_name, move_file_name)

            elif image_ext == '.depth.png':
                move_file_name = new_file_name + 'depth/' + np.str(object) + '_' + np.str(
                    image_num) + '_depth.png'
                if idx == 0:
                    print(f'Old file: {old_file_name}')
                    print(f'New file: {move_file_name}')
                shutil.copyfile(old_file_name, move_file_name)

            elif image_ext == '.aff.png':
                move_file_name = new_file_name + 'masks/' + np.str(object) + '_' + np.str(
                    image_num) + '_label.png'
                if idx == 0:
                    print(f'Old file: {old_file_name}')
                    print(f'New file: {move_file_name}')
                shutil.copyfile(old_file_name, move_file_name)

            else:
                assert "*** IMAGE EXT DOESN'T EXIST ***"

        ###############
        ###############
        if image_ext == '.aff.png':
            offset_train += len(train_files)
            offset_val   += len(val_files)
            offset_test  += len(test_files)