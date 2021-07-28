import numpy as np
import torch

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).parent.absolute().resolve(strict=True)
ROOT_DIR_PATH = str(ROOT_DIR_PATH) + '/'

'''
Framework Selection:
'MaskRCNN'
'AffNet'
'''

# Prelim for naming experiment.
FRAMEWORK           = 'AffNet'
EXP_DATASET_NAME    = 'ARLAffPose_Real_RGB'
EXP_NUM             = 'v0'

'''
Backbone Selection:
'resnet50'
'resnet18'
'''

# TODO: add VGG-16.
BACKBONE_FEAT_EXTRACTOR = 'resnet50'

IS_PRETRAINED = True
RESNET_PRETRAINED_WEIGHTS = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
MASKRCNN_PRETRAINED_WEIGHTS = 'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth'  # resnet50
# MASKRCNN_PRETRAINED_WEIGHTS = ROOT_DIR_PATH + 'pretrained_coco_weights_ResNet18.pth' # resnet18
# MASKRCNN_PRETRAINED_WEIGHTS = ROOT_DIR_PATH + 'pretrained_coco_weights_ResNet50.pth' # resnet50

RESTORE_MASKRCNN_WEIGHTS = ROOT_DIR_PATH + 'trained_models/ARLAffPose_Real_RGB/MaskRCNN_ARLAffPose_Real_RGB_640x640_v0/BEST_MODEL.pth'
RESTORE_AFFNET_WEIGHTS = ROOT_DIR_PATH + 'trained_models/ARLAffPose_Real_RGB/AffNet_ARLAffPose_Real_RGB_640x640_v0/BEST_MODEL.pth'

''' 
MaskRCNN configs. 
see reference here https://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/
'''

# Used to threshold predictions based on objectiveness.
CONFIDENCE_THRESHOLD = 0.7

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
ROIALIGN_MASK_OUTPUT_SIZE = (14, 14)  # todo (ak): try (128, 128) like AffNet
ROIALIGN_SAMPLING_RATIO = 2

# RoIHeads parameters
BOX_FG_IOU_THRESH = 0.5
BOX_BG_IOU_THRESH = 0.5
BOX_NUM_SAMPLES = 512
BOX_POSITIVE_FRACTION = 0.25
BOX_REG_WEIGHTS = (10., 10., 5., 5.)
BOX_SCORE_THRESH = 0.1
BOX_NMS_THRESH = 0.6
BOX_NUM_DETECTIONS = 100


'''
ARL AffPose Configs.
'''

ROOT_DATA_PATH = '/data/Akeaveny/Datasets/ARLAffPose/'
SELECT_EVERY_ITH_FRAME = 3  # similar to YCB-Video Dat

NUM_CLASSES = 11 + 1
NUM_OBJECT_CLASSES = 11 + 1  # 1 is for the background
NUM_AFF_CLASSES = 9 + 1  # 1 is for the background

IMAGE_MEAN = [115.16123185/255, 94.20813919/255, 84.34889709/255]
IMAGE_STD = [56.62171952/255, 56.86680141/255, 36.95978531/255]
RESIZE = (int(1280/1), int(720/1))
CROP_SIZE = (int(640), int(640))

DATA_DIRECTORY = ROOT_DATA_PATH + 'Real/'
DATA_DIRECTORY_TRAIN = DATA_DIRECTORY + 'train/'
DATA_DIRECTORY_VAL = DATA_DIRECTORY + 'val/'
DATA_DIRECTORY_TEST = ROOT_DATA_PATH + 'Real/' + 'test/'

SYN_DATA_DIRECTORY = ROOT_DATA_PATH + 'Syn/'
SYN_DATA_DIRECTORY_TRAIN = SYN_DATA_DIRECTORY + 'train/'
SYN_DATA_DIRECTORY_VAL = SYN_DATA_DIRECTORY + 'val/'
SYN_DATA_DIRECTORY_TEST = SYN_DATA_DIRECTORY + 'test/'

IMG_SIZE = str(CROP_SIZE[0]) + 'x' + str(CROP_SIZE[1])

'''
Hyperparams.
'''

# train on the GPU or on the CPU, if a GPU is not available
CPU_DEVICE = 'cpu'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("using device: {} ..".format(DEVICE))

RANDOM_SEED = 1234

NUM_EPOCHS = 10
BATCH_SIZE  = 1
NUM_WORKERS = 1

LEARNING_RATE = 1e-03
WEIGHT_DECAY = 1e-04
MOMENTUM = 0.9

''' 
Configs for logging & eval.
'''

# Logging.
EXP_NAME = FRAMEWORK + '_' + EXP_DATASET_NAME + '_' + IMG_SIZE + '_' + EXP_NUM
TRAINED_MODELS_DIR = str(ROOT_DIR_PATH) + 'trained_models/' + EXP_DATASET_NAME + '/' + EXP_NAME
TEST_SAVE_FOLDER = DATA_DIRECTORY_TEST + 'pred_' + EXP_NAME + '/'
MODEL_SAVE_PATH = str(TRAINED_MODELS_DIR) + '/'
BEST_MODEL_SAVE_PATH = MODEL_SAVE_PATH + 'BEST_MODEL.pth'

# Eval.
MATLAB_SCRIPTS_DIR = np.str(ROOT_DIR_PATH +'matlab/')
OBJ_EVAL_SAVE_FOLDER = DATA_DIRECTORY_TEST + 'pred_obj/'
AFF_EVAL_SAVE_FOLDER = DATA_DIRECTORY_TEST + 'pred_aff/'

NUM_TEST = 250
NUM_EVAL = 250
TEST_GT_EXT = "_gt.png"
TEST_PRED_EXT = "_pred.png"
TEST_OBJ_PART_EXT = "_obj_part.png"