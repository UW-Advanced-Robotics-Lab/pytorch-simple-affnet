import numpy as np
import torch

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).parent.absolute().resolve(strict=True)
ROOT_DIR_PATH = str(ROOT_DIR_PATH) + '/'

#######################################
#######################################

'''
FRAMEWORK Selection:
'MaskRCNN'
'''

# TODO: prelim for naming
FRAMEWORK           = 'MaskRCNN'
EXP_DATASET_NAME    = 'ARLVicon_Real_RGB'
EXP_NUM             = 'v1_Test_Resize_640'

#######################################
#######################################

'''
BACKBONE Selection:
'resnet50'
'resnet18'
'''

BACKBONE_FEAT_EXTRACTOR = 'resnet18'

IS_PRETRAINED = True
RESNET_PRETRAINED_WEIGHTS = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
MASKRCNN_PRETRAINED_WEIGHTS = ROOT_DIR_PATH + 'pretrained_coco_weights.pth' # 'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth'

# RESTORE_TRAINED_WEIGHTS = '/data/Akeaveny/weights/AffNet/UMD_Real_RGB/MaskRCNN_UMD_Real_RGB_384x384_v4_Test_Resize_384/BEST_MODEL.pth'
RESTORE_TRAINED_WEIGHTS = '/home/akeaveny/git/PyTorch-Simple-AffNet/snapshots/ARLVicon_Real_RGB/MaskRCNN_ARLVicon_Real_RGB_640x640_v0_Test_Resize_640/BEST_MODEL.pth'

#######################################
#######################################

# train on the GPU or on the CPU, if a GPU is not available
CPU_DEVICE = 'cpu'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("using device: {} ..".format(DEVICE))

RANDOM_SEED = 1234

NUM_EPOCHS = 30

NUM_REAL_IMAGES = int(20e3) # 20190
NUM_TRAIN = int(np.floor(0.7*NUM_REAL_IMAGES))
NUM_VAL   = int(np.floor(0.3*NUM_REAL_IMAGES))

NUM_STEPS      = int(NUM_EPOCHS*NUM_TRAIN) # ~30 epochs at 5000 images/epoch
NUM_VAL_STEPS  = int(NUM_EPOCHS*NUM_VAL)   # ~30 epochs at 1250 images/epoch

BATCH_SIZE  = 1
NUM_WORKERS = 2

#######################################
#######################################

LEARNING_RATE = 1e-03
WEIGHT_DECAY = 1e-04

######################################
#######################################
''' MaskRCNN configs '''
# see https://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/

CONFIDENCE_THRESHOLD = 0.35 # TORCHVISION: 0.4 or SIMPLE:0.35

# Anchor Generator
ANCHOR_SIZES = (16, 32, 64)
ANCHOR_RATIOS = (0.5, 1, 1.5)

# transform parameters
# MIN_SIZE = 800
# MAX_SIZE = 1333
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# RPN parameters
RPN_FG_IOU_THRESH = 0.7
RPN_BG_IOU_THRESH = 0.3
RPN_NUM_SAMPLES = 256
RPN_POSITIVE_FRACTION = 0.5
RPN_REG_WEIGHTS = (1., 1., 1., 1.)
RPN_PRE_NMS_TOP_N_TRAIN = 2000
RPN_PRE_NMS_TOP_N_TEST = 1000
RPN_POST_NMS_TOP_N_TRAIN = 2000
RPN_POST_NMS_TOP_N_TEST = 1000
RPN_NMS_THRESH = 0.7

# RoIAlign parameters
ROIALIGN_BOX_OUTPUT_SIZE = (7, 7)
ROIALIGN_MASK_OUTPUT_SIZE = (14, 14) # todo (ak): try (128, 128) like AffNet
ROIALIGN_SAMPLING_RATIO = 2

# RoIHeads parameters
BOX_FG_IOU_THRESH = 0.5
BOX_BG_IOU_THRESH = 0.5
BOX_NUM_SAMPLES = 512
BOX_POSITIVE_FRACTION = 0.25
BOX_REG_WEIGHTS = (10., 10., 5., 5.)
BOX_SCORE_THRESH = 0.1
BOX_NMS_THRESH = 0.6
BOX_NUM_DETECTIONS = 100             # todo: change from default

#######################################
### COCO
#######################################

# COCO_ROOT_DATA_PATH = '/data/Akeaveny/Datasets/COCO/'
# COCO_TRAIN_SPLIT = 'train2017'
# COCO_VAL_SPLIT = 'val2017'
#
# COCO_NUM_CLASSES = 79 + 1

#######################################
### UMD
#######################################

# ROOT_DATA_PATH = '/data/Akeaveny/Datasets/UMD/'
#
# NUM_CLASSES = 7 + 1
# NUM_OBJECT_CLASSES = 17 + 1         # 1 is for the background
# NUM_AFF_CLASSES = 7 + 1         # 1 is for the background
#
# ### REAL
# DATA_DIRECTORY = ROOT_DATA_PATH + 'Real/'
# DATA_DIRECTORY_TRAIN = DATA_DIRECTORY + 'train/'
# DATA_DIRECTORY_VAL = DATA_DIRECTORY + 'val/'
# DATA_DIRECTORY_TEST = DATA_DIRECTORY + 'test/'
#
# IMAGE_MEAN   = [98.92739272/255, 66.78827961/255, 71.00867078/255]
# IMAGE_STD    = [26.53540375/255, 31.51117582/255, 31.75977128/255]
# RESIZE       = (int(640/1), int(480/1))
# CROP_SIZE   = (int(384), int(384))
# MIN_SIZE = MAX_SIZE = 384
#
# ### SYN
# # DATA_DIRECTORY = ROOT_DATA_PATH + 'Syn/'
# # DATA_DIRECTORY_TRAIN = DATA_DIRECTORY + 'train/'
# # DATA_DIRECTORY_VAL = DATA_DIRECTORY + 'val/'
# # DATA_DIRECTORY_TEST = DATA_DIRECTORY + 'test/'
#
# # IMAGE_MEAN   = [135.4883242/255, 143.06856056/255, 125.6341276/255]
# # IMAGE_STD    = [39.76640244/255, 46.91340711/255,  46.25064666/255]
# # RESIZE       = (int(640/1), int(480/1))
# # CROP_SIZE   = (int(384), int(384))
# # MIN_SIZE = MAX_SIZE = 384
#
# IMG_SIZE = str(CROP_SIZE[0]) + 'x' + str(CROP_SIZE[1])

#######################################
### Elavator
#######################################

# ROOT_DATA_PATH = '/data/Akeaveny/Datasets/Elevator/'
#
# NUM_CLASSES = 1 + 1
#
# ### REAL
# DATA_DIRECTORY = ROOT_DATA_PATH + 'Real/'
# DATA_DIRECTORY_TRAIN = DATA_DIRECTORY + 'train/'
# DATA_DIRECTORY_VAL = DATA_DIRECTORY + 'val/'
# DATA_DIRECTORY_TEST = DATA_DIRECTORY + 'test/'
#
# RESIZE       = (int(672/1), int(376/1))
# CROP_SIZE   = (int(384), int(384))
# MIN_SIZE = MAX_SIZE = 384
#
# IMG_SIZE = str(CROP_SIZE[0]) + 'x' + str(CROP_SIZE[1])

#######################################
### ARL VICON
#######################################

ROOT_DATA_PATH = '/data/Akeaveny/Datasets/ARLVicon/'

NUM_CLASSES = 1 + 1
NUM_OBJECT_CLASSES = 1 + 1      # 1 is for the background
NUM_AFF_CLASSES = 2 + 1         # 1 is for the background

### Syn
# DATA_DIRECTORY = ROOT_DATA_PATH + 'Real/'
# DATA_DIRECTORY = ROOT_DATA_PATH + 'Syn/'
DATA_DIRECTORY = ROOT_DATA_PATH + 'RealandSyn/'
DATA_DIRECTORY_TRAIN = DATA_DIRECTORY + 'train/'
DATA_DIRECTORY_VAL = DATA_DIRECTORY + 'val/'
DATA_DIRECTORY_TEST = DATA_DIRECTORY + 'test/'

RESIZE       = (int(1280/1), int(720/1))
CROP_SIZE   = (int(640), int(640))
MIN_SIZE = MAX_SIZE = 640

IMG_SIZE = str(CROP_SIZE[0]) + 'x' + str(CROP_SIZE[1])

#######################################
### ARL AffPose
#######################################

# ROOT_DATA_PATH = '/data/Akeaveny/Datasets/ARLAffPose/'
#
# NUM_CLASSES = 11 + 1
# NUM_OBJECT_CLASSES = 11 + 1     # 1 is for the background
# NUM_AFF_CLASSES = 9 + 1         # 1 is for the background
#
# ### REAL
# # DATA_DIRECTORY = ROOT_DATA_PATH + 'Real/'
# # DATA_DIRECTORY_TRAIN = DATA_DIRECTORY + 'train/'
# # DATA_DIRECTORY_VAL = DATA_DIRECTORY + 'val/'
# # DATA_DIRECTORY_TEST = DATA_DIRECTORY + 'test/'
#
# ### SYN
# DATA_DIRECTORY = ROOT_DATA_PATH + 'Syn/'
# DATA_DIRECTORY_TRAIN = DATA_DIRECTORY + 'train/'
# DATA_DIRECTORY_VAL = DATA_DIRECTORY + 'val/'
# DATA_DIRECTORY_TEST = DATA_DIRECTORY + 'test/'
#
# RESIZE       = (int(1280/1), int(720/1))
# CROP_SIZE   = (int(640), int(640))
# MIN_SIZE, MAX_SIZE = 480, 640
#
# IMG_SIZE = str(CROP_SIZE[0]) + 'x' + str(CROP_SIZE[1])

#######################################
#######################################
''' PRELIM FOR SAVED WEIGHTS'''

### TEST
MATLAB_SCRIPTS_DIR = np.str(ROOT_DIR_PATH +'/matlab/')
EVAL_SAVE_FOLDER = DATA_DIRECTORY_TEST + 'pred/'

NUM_TEST = 50
TEST_GT_EXT = "_gt.png"
TEST_PRED_EXT = "_pred.png"

EXP_NAME = FRAMEWORK + '_' + EXP_DATASET_NAME + '_' + IMG_SIZE + '_' + EXP_NUM
SNAPSHOT_DIR = str(ROOT_DIR_PATH) + 'snapshots/' + EXP_DATASET_NAME + '/' + EXP_NAME
TEST_SAVE_FOLDER = DATA_DIRECTORY_TEST + 'pred_' + EXP_NAME + '/'

MODEL_SAVE_PATH = str(SNAPSHOT_DIR) + '/'
BEST_MODEL_SAVE_PATH = MODEL_SAVE_PATH + 'BEST_MODEL.pth'


