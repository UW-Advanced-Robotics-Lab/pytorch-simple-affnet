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
FRAMEWORK = 'AffNet'
EXP_DATASET = 'ARLAffPose'
EXP_DOMAIN = 'Real'
EXP_IMAGES = 'RGB'
EXP_NUM = 'v3_syn_frozen_backbone'

'''
Backbone Selection:
'resnet50'
'resnet18'
'''

BACKBONE_FEAT_EXTRACTOR = 'resnet50'

IS_PRETRAINED = True
RESNET_PRETRAINED_WEIGHTS = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
MASKRCNN_PRETRAINED_WEIGHTS = 'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth'  # resnet50

RESTORE_UMD_AFFNET_WEIGHTS = '/data/Akeaveny/weights/AffNet/UMD/UMD_Real_RGB/AffNet_UMD_Real_RGB_640x640_v4_imgaug/BEST_MODEL.pth'

RESTORE_ARL_TORCHVISION_MASKRCNN_WEIGHTS = ROOT_DIR_PATH + 'trained_models/ARL_AffPose_Real_RGB/MaskRCNN_ARLAffPose_Real_RGB_640x640_v0_torchvision/maskrcnn_epoch_4.pth'

RESTORE_ARL_MASKRCNN_WEIGHTS = '/data/Akeaveny/weights/AffNet/ARLAffPose/MaskRCNN/MaskRCNN_ARLAffPose_Real_RGB_640x640_v1_transpose_conv2d_28x28/BEST_MODEL.pth'
RESTORE_SYN_ARL_MASKRCNN_WEIGHTS = '/data/Akeaveny/weights/AffNet/ARLAffPose/MaskRCNN/MaskRCNN_ARLAffPose_Syn_RGB_640x640_v3_syn_frozen_backbone/BEST_MODEL.pth'
RESTORE_SYN_AND_REAL_ARL_MASKRCNN_WEIGHTS = '/data/Akeaveny/weights/AffNet/ARLAffPose/MaskRCNN/MaskRCNN_ARLAffPose_Real_and_Syn_RGB_640x640_v3_syn_frozen_backbone/BEST_MODEL.pth'

RESTORE_ARL_AFFNET_WEIGHTS = '/data/Akeaveny/weights/AffNet/ARLAffPose/AffNet/AffNet_ARLAffPose_Real_RGB_640x640_v1_transpose_conv2d_28x28/BEST_MODEL.pth'
RESTORE_SYN_ARL_AFFNET_WEIGHTS = '/data/Akeaveny/weights/AffNet/ARLAffPose/AffNet/AffNet_ARLAffPose_Syn_RGB_640x640_v3_syn_frozen_backbone/affnet_epoch_1.pth'
RESTORE_SYN_AND_REAL_ARL_AFFNET_WEIGHTS = '/data/Akeaveny/weights/AffNet/ARLAffPose/AffNet/AffNet_ARLAffPose_Real_and_Syn_RGB_640x640_v3_syn_frozen_backbone/BEST_MODEL.pth'

''' 
MaskRCNN configs. 
see reference here https://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/
'''

# Used to threshold predictions based on objectiveness.
OBJ_CONFIDENCE_THRESHOLD = 0.7
MASK_THRESHOLD = 0.5

# Anchor Generator
ANCHOR_SIZES = (32, 64, 128, 256, 384)
ANCHOR_RATIOS = (0.5, 1, 2)

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
ROIALIGN_MASK_OUTPUT_SIZE = (14, 14)  # (7, 7) or (14, 14)
ROIALIGN_SAMPLING_RATIO = 2

# RoIHeads parameters
BOX_FG_IOU_THRESH = 0.5
BOX_BG_IOU_THRESH = 0.5
BOX_NUM_SAMPLES = 512
BOX_POSITIVE_FRACTION = 0.25
BOX_REG_WEIGHTS = (10., 10., 5., 5.)
BOX_SCORE_THRESH = 0.1
BOX_NMS_THRESH = 0.6
BOX_NUM_DETECTIONS = 10  # 200

'''
COCO Configs.
'''

COCO_ROOT_DATA_PATH = '/data/Akeaveny/Datasets/COCO/'
COCO_TRAIN_SPLIT = 'train2017'
COCO_VAL_SPLIT = 'val2017'

COCO_NUM_CLASSES = 79 + 1

COCO_TEST_SAVE_FOLDER = COCO_ROOT_DATA_PATH + 'test/'
COCO_EVAL_SAVE_FOLDER = COCO_ROOT_DATA_PATH + 'pred/'

'''
UMD Configs.
'''

UMD_ROOT_DATA_PATH = '/data/Akeaveny/Datasets/UMD/'

UMD_NUM_CLASSES = 17 + 1
UMD_NUM_OBJECT_CLASSES = 17 + 1  # 1 is for the background
UMD_NUM_AFF_CLASSES = 7 + 1  # 1 is for the background

UMD_IMAGE_MEAN = [148.06817006/255, 175.72064619/255, 164.09241116]
UMD_IMAGE_STD = [19.12525118/255, 34.56108673/255, 30.58577597]
UMD_RESIZE = (int(480), int(640))  # (int(640), int(480))
UMD_CROP_SIZE = (int(480), int(640))  # (int(640), int(480))
UMD_MIN_SIZE = 600
UMD_MAX_SIZE = 1000

UMD_DATA_DIRECTORY = UMD_ROOT_DATA_PATH + 'Real/'
UMD_DATA_DIRECTORY_TRAIN = UMD_DATA_DIRECTORY + 'train/'
UMD_DATA_DIRECTORY_VAL = UMD_DATA_DIRECTORY + 'val/'
UMD_DATA_DIRECTORY_TEST = UMD_DATA_DIRECTORY + 'test/'

# UMD_SYN_DATA_DIRECTORY = UMD_ROOT_DATA_PATH + 'Syn/'
# UMD_SYN_DATA_DIRECTORY_TRAIN = UMD_SYN_DATA_DIRECTORY + 'train/'
# UMD_SYN_DATA_DIRECTORY_VAL = UMD_SYN_DATA_DIRECTORY + 'val/'
# UMD_SYN_DATA_DIRECTORY_TEST = UMD_SYN_DATA_DIRECTORY + 'test/'

UMD_TEST_SAVE_FOLDER = UMD_DATA_DIRECTORY_TEST + 'test/'
UMD_AFF_EVAL_SAVE_FOLDER = UMD_DATA_DIRECTORY_TEST + 'pred_aff/'

IMG_SIZE = str(UMD_CROP_SIZE[0]) + 'x' + str(UMD_CROP_SIZE[1])

'''
ARL AffPose Configs.
'''

ARL_ROOT_DATA_PATH = '/data/Akeaveny/Datasets/ARLAffPose/'

ARL_NUM_CLASSES = 11 + 1
ARL_NUM_OBJECT_CLASSES = 11 + 1  # 1 is for the background
ARL_NUM_AFF_CLASSES = 9 + 1  # 1 is for the background

ARL_IMAGE_MEAN = [115.16123185/255, 94.20813919/255, 84.34889709/255]
ARL_IMAGE_STD = [56.62171952/255, 56.86680141/255, 36.95978531/255]
ARL_RESIZE = (int(1280/1), int(720/1))
ARL_CROP_SIZE = (int(640), int(640))

ARL_DATA_DIRECTORY = ARL_ROOT_DATA_PATH + 'Real/'
ARL_DATA_DIRECTORY_TRAIN = ARL_DATA_DIRECTORY + 'train/'
ARL_DATA_DIRECTORY_VAL = ARL_DATA_DIRECTORY + 'val/'
ARL_DATA_DIRECTORY_TEST = ARL_ROOT_DATA_PATH + 'Real/' + 'test/'
ARL_DATA_DIRECTORY_WAM = ARL_ROOT_DATA_PATH + 'WAM/' + 'test/'

ARL_SYN_DATA_DIRECTORY = ARL_ROOT_DATA_PATH + 'Syn/'
ARL_SYN_DATA_DIRECTORY_TRAIN = ARL_SYN_DATA_DIRECTORY + 'train/'
ARL_SYN_DATA_DIRECTORY_VAL = ARL_SYN_DATA_DIRECTORY + 'val/'
ARL_SYN_DATA_DIRECTORY_TEST = ARL_SYN_DATA_DIRECTORY + 'test/'

ARL_TEST_SAVE_FOLDER = ARL_DATA_DIRECTORY_TEST + 'test/'
ARL_OBJ_EVAL_SAVE_FOLDER = ARL_DATA_DIRECTORY_TEST + 'pred_obj/'
ARL_AFF_EVAL_SAVE_FOLDER = ARL_DATA_DIRECTORY_TEST + 'pred_aff/'

ARL_IMG_SIZE = str(ARL_CROP_SIZE[0]) + 'x' + str(ARL_CROP_SIZE[1])

'''
YCB Video Configs.
'''

YCB_DATASET_ROOT_PATH = '/data/Akeaveny/Datasets/YCB_Affordance_Dataset'
YCB_IMAGE_DOMAIN = 'Real'

DENSEFUSION_ROOT_PATH = '/home/akeaveny/git/DenseFusion/'
YCB_TRAIN_FILE = DENSEFUSION_ROOT_PATH + 'datasets/ycb/dataset_config/train_data_list.txt'
YCB_TEST_FILE = DENSEFUSION_ROOT_PATH + 'datasets/ycb/dataset_config/test_data_list.txt'

YCB_NUM_CLASSES = 21 + 1
YCB_NUM_OBJECT_CLASSES = 21 + 1  # 1 is for the background
YCB_NUM_AFF_CLASSES = 7 + 1  # 1 is for the background

YCB_IMAGE_MEAN = [115.16123185/255, 94.20813919/255, 84.34889709/255]
YCB_IMAGE_STD = [56.62171952/255, 56.86680141/255, 36.95978531/255]
YCB_RESIZE = (int(640), int(640))
YCB_CROP_SIZE = (int(640), int(640))

YCB_TEST_SAVE_FOLDER = YCB_DATASET_ROOT_PATH + '/test/'
YCB_OBJ_EVAL_SAVE_FOLDER = YCB_DATASET_ROOT_PATH + '/pred_obj/'
YCB_AFF_EVAL_SAVE_FOLDER = YCB_DATASET_ROOT_PATH + '/pred_aff/'

'''
Hyperparams.
'''

# train on the GPU or on the CPU, if a GPU is not available
CPU_DEVICE = 'cpu'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("using device: {} ..".format(DEVICE))

RANDOM_SEED = 1234

NUM_EPOCHS = 20
EPOCH_TO_TRAIN_FULL_DATASET = 0
BATCH_SIZE = 1
NUM_WORKERS = 4

NUM_IMAGES_PER_EPOCH = 500
NUM_TRAIN = int(NUM_IMAGES_PER_EPOCH*0.8)
NUM_VAL = int(NUM_IMAGES_PER_EPOCH-NUM_TRAIN)
NUM_TEST = 250
NUM_EVAL = 250

CLIP_GRADIENT = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
MILESTONES = [3, 5]
GAMMA = 0.1

NUM_CLASSES = UMD_NUM_CLASSES
NUM_OBJECT_CLASSES = UMD_NUM_OBJECT_CLASSES
NUM_AFF_CLASSES = UMD_NUM_AFF_CLASSES

MIN_SIZE = UMD_MIN_SIZE
MAX_SIZE = UMD_MAX_SIZE
IMAGE_MEAN = UMD_IMAGE_MEAN
IMAGE_STD = UMD_IMAGE_STD

''' 
Configs for logging & eval.
'''

# Logging.
EXP_DATASET_NAME = f'{EXP_DATASET}_{EXP_DOMAIN}_{EXP_IMAGES}'
EXP_NAME = FRAMEWORK + '_' + EXP_DATASET_NAME + '_' + ARL_IMG_SIZE + '_' + EXP_NUM
TRAINED_MODELS_DIR = str(ROOT_DIR_PATH) + 'trained_models/' + EXP_DATASET_NAME + '/' + EXP_NAME
MODEL_SAVE_PATH = str(TRAINED_MODELS_DIR) + '/'
BEST_MODEL_SAVE_PATH = MODEL_SAVE_PATH + 'BEST_MODEL.pth'

# Eval.
MATLAB_SCRIPTS_DIR = np.str(ROOT_DIR_PATH + 'matlab/')

TEST_GT_EXT = "_gt.png"
TEST_PRED_EXT = "_pred.png"
TEST_OBJ_PART_EXT = "_obj_part.png"
