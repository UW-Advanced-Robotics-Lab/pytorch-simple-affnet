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
EXP_DATASET_NAME    = 'UMD_Real_RGB'
EXP_NUM             = 'v2_AffNet_128x128_VGG16_No_Warmup'

#######################################
#######################################

'''
BACKBONE Selection:
'resnet50'
'''

BACKBONE_FEAT_EXTRACTOR = 'resnet50'

IS_TRAIN_WITH_DEPTH = False
NUM_CHANNELS        = 3     # RGB=3, DEPTH=1 or RGB=4
NUM_RGB_CHANNELS    = 3
NUM_D_CHANNELS      = 3

OUTPUT_STRIDE = 16

IS_PRETRAINED = False
RESNET_PRETRAINED_WEIGHTS = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
MASKRCNN_PRETRAINED_WEIGHTS = 'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth'

# RESTORE_TRAINED_WEIGHTS = '/home/akeaveny/git/PyTorch-Simple-MaskRCNN/snapshots/UMD_Real_RGB/MaskRCNN_UMD_Real_RGB_128x128_v0_Torchvision_MaskRCNN_128x128/BEST_MODEL.pth'
# RESTORE_TRAINED_WEIGHTS = '/home/akeaveny/git/PyTorch-Simple-MaskRCNN/snapshots/UMD_Real_RGB/MaskRCNN_UMD_Real_RGB_128x128_v0_Simple_MaskRCNN_128x128/BEST_MODEL.pth'
# RESTORE_TRAINED_WEIGHTS = '/home/akeaveny/git/PyTorch-Simple-MaskRCNN/snapshots/UMD_Real_RGB/MaskRCNN_UMD_Real_RGB_128x128_v0_AffNet_128x128/BEST_MODEL.pth'
RESTORE_TRAINED_WEIGHTS = '/home/akeaveny/git/PyTorch-Simple-MaskRCNN/snapshots/UMD_Syn_RGB/MaskRCNN_UMD_Syn_RGB_128x128_v0_AffNet_128x128/BEST_MODEL.pth'

#######################################
#######################################

# train on the GPU or on the CPU, if a GPU is not available
CPU_DEVICE = 'cpu'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("using device: {} ..".format(DEVICE))

RANDOM_SEED = 1234

NUM_EPOCHS = 5

NUM_TRAIN  = 5000
NUM_VAL    = 1250

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

CONFIDENCE_THRESHOLD = 0.5 # TORCHVISION: 0.4 or SIMPLE:0.35

# Anchor Generator
ANCHOR_SIZES = (16, 32, 64)
ANCHOR_RATIOS = (0.5, 1, 1.5)

# transform parameters
MIN_SIZE = 800
MAX_SIZE = 1333
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
### PennFudan
#######################################

# ROOT_DATASET_PATH = '/data/Akeaveny/Datasets/PennFudanPed'
# NUM_CLASSES = 2

#######################################
### UMD
#######################################
''' DATASET PRELIMS'''

ROOT_DATA_PATH = '/data/Akeaveny/Datasets/UMD/'

NUM_CLASSES = 7 + 1
NUM_OBJECT_CLASSES = 17 + 1         # 1 is for the background
NUM_AFF_CLASSES = 7 + 1         # 1 is for the background

#######################################
#######################################

### REAL
DATA_DIRECTORY = ROOT_DATA_PATH + 'Real/'
DATA_DIRECTORY_TRAIN = DATA_DIRECTORY + 'train/'
DATA_DIRECTORY_VAL = DATA_DIRECTORY + 'val/'
DATA_DIRECTORY_TEST = DATA_DIRECTORY + 'test/'

# IMG_MEAN   = [98.92739272, 66.78827961, 71.00867078, 135.8963934]
# IMG_STD    = [26.53540375, 31.51117582, 31.75977128, 38.23637208]
# RESIZE     = (int(640/3), int(480/3))
# INPUT_SIZE = (int(128), int(128))

### PR
# DATA_DIRECTORY = ROOT_DATA_PATH + 'PR/'
# DATA_DIRECTORY_TRAIN = DATA_DIRECTORY + 'train/'
# DATA_DIRECTORY_VAL = DATA_DIRECTORY + 'val/'
# DATA_DIRECTORY_TEST = DATA_DIRECTORY + 'test/'

# PR_IMG_MEAN   = [138.58907803, 151.88310081, 125.22561575, 130.00560063]
# PR_IMG_STD    = [30.37894422, 38.44065602, 43.2841762, 43.57943909]
# PR_RESIZE     = (int(640*1.35/3), int(480*1.35/3))
# PR_INPUT_SIZE = (int(128), int(128))

### DR
# DATA_DIRECTORY_DR = ROOT_DATA_PATH + 'DR/'
# DATA_DIRECTORY_TRAIN = DATA_DIRECTORY + 'train/'
# DATA_DIRECTORY_VAL = DATA_DIRECTORY + 'val/'
# DATA_DIRECTORY_TEST = DATA_DIRECTORY + 'test/'

# DR_IMG_MEAN   = [134.38601217, 137.02879418, 129.27239013, 140.01491372]
# DR_IMG_STD    = [48.88474747, 54.86081706, 48.8507932, 32.20115424]
# DR_RESIZE     = (int(640/3), int(480/3))
# DR_INPUT_SIZE = (int(128), int(128))


### SYN
# DATA_DIRECTORY = ROOT_DATA_PATH + 'Syn/'
# DATA_DIRECTORY_TRAIN = DATA_DIRECTORY + 'train/'
# DATA_DIRECTORY_VAL = DATA_DIRECTORY + 'val/'
# DATA_DIRECTORY_TEST = DATA_DIRECTORY + 'test/'

IMG_MEAN   = [135.4883242,  143.06856056, 125.6341276, 134.57706755]
IMG_STD    = [39.76640244, 46.91340711, 46.25064666, 38.62958981]
RESIZE     = (int(640/3), int(480/3))
INPUT_SIZE = (int(128), int(128))

IMG_SIZE = str(INPUT_SIZE[0]) + 'x' + str(INPUT_SIZE[1])

#######################################
#######################################
''' PRELIM FOR SAVED WEIGHTS'''

### TEST
MATLAB_SCRIPTS_DIR = np.str(ROOT_DIR_PATH +'/matlab/')
EVAL_SAVE_FOLDER = DATA_DIRECTORY_TEST + 'pred/'

NUM_TEST = 100
TEST_GT_EXT = "_gt.png"
TEST_PRED_EXT = "_pred.png"

EVAL_UPDATE         = int(NUM_STEPS/150) # eval model every thousand iterations
TENSORBOARD_UPDATE  = int(NUM_STEPS/150)
SAVE_PRED_EVERY     = int(NUM_STEPS/NUM_EPOCHS*5)

EXP_NAME = FRAMEWORK + '_' + EXP_DATASET_NAME + '_' + IMG_SIZE + '_' + EXP_NUM
SNAPSHOT_DIR = str(ROOT_DIR_PATH) + 'snapshots/' + EXP_DATASET_NAME + '/' + EXP_NAME
TEST_SAVE_FOLDER = DATA_DIRECTORY_TEST + 'pred_' + EXP_NAME + '/'

MODEL_SAVE_PATH = str(SNAPSHOT_DIR) + '/'
BEST_MODEL_SAVE_PATH = MODEL_SAVE_PATH + 'BEST_MODEL.pth'


