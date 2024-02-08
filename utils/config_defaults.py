"""Configs."""
from fvcore.common.config import CfgNode


_C = CfgNode()


# Output basedir.
_C.OUTPUT_DIR = "logs"
_C.RNG_SEED = 6293
_C.DIST_BACKEND = "nccl"


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()
_C.DATA.DATASET = "IHC"
_C.DATA.DATASET_NAME = ["IHC"]
# _C.DATA.DATASET_NAME = ["IHC", "GraphLoc", "MSTLoc", "laceDNN"]
_C.DATA.PATH_TO_DATA_DIR = "/root/autodl-tmp/dataset"
_C.DATA.RESULT_DIR = "."
# _C.DATA.RESULT_DIR = "/root/autodl-tmp"

_C.DATA.MEAN = [0.485, 0.456, 0.406]  # The mean value of pixels across the R G B channels.
_C.DATA.STD = [0.229, 0.224, 0.225]  # The std value of pixels across the R G B channels.
_C.DATA.CROP_SIZE = 2200  # Image size after cropping


# ---------------------------------------------------------------------------- #
# SAE options.
# ---------------------------------------------------------------------------- #
_C.SAE = CfgNode()
_C.SAE.MODEL_NAME = "2880-480-128"
_C.SAE.CONSTRUCT = False
_C.SAE.TRAIN = 0
_C.SAE.CKP = True
_C.SAE.BASE_LR = [1e-3, 1e-3, 1e-3]
_C.SAE.END_LR = 1e-8
_C.SAE.LOSS_FUNC = "huber"
_C.SAE.BETA = 0.01
_C.SAE.T0 = 10
_C.SAE.N_T = 0.5
_C.SAE.EPOCH_NUM = 90
_C.SAE.ACCUMULATION_STEPS = 10
_C.SAE.EVALUATION_STEPS = 5
_C.SAE.PRINT_STEPS = 20

# ---------------------------------------------------------------------------- #
# Classifier options.
# ---------------------------------------------------------------------------- #
_C.CLASSIFIER = CfgNode()

_C.CLASSIFIER.CONSTRUCT = True
_C.CLASSIFIER.PRETRAIN = False
_C.CLASSIFIER.PRE = CfgNode()
_C.CLASSIFIER.PRE.CKP = False
_C.CLASSIFIER.PRE.BASE_LR = 2e-5
_C.CLASSIFIER.PRE.END_LR = 1e-8
_C.CLASSIFIER.PRE.END_SCALE = 5e-4
_C.CLASSIFIER.PRE.LOSS_FUNC = "nt_xent"
_C.CLASSIFIER.PRE.T0 = 10
_C.CLASSIFIER.PRE.N_T = 0.5
_C.CLASSIFIER.PRE.EPOCH_NUM = 120
_C.CLASSIFIER.PRE.ACCUMULATION_STEPS = 5
_C.CLASSIFIER.PRE.EVALUATION_STEPS = 5
_C.CLASSIFIER.PRE.PRINT_STEPS = 20
_C.CLASSIFIER.PRE.M = 0.99
_C.CLASSIFIER.PRE.TEMPERATURE = 0.15

_C.CLASSIFIER.TRAIN = True
_C.CLASSIFIER.CKP = False
_C.CLASSIFIER.BASE_LR = 5e-5
_C.CLASSIFIER.HEAD_BASE_LR = 5e-5
_C.CLASSIFIER.END_LR = 1e-8
_C.CLASSIFIER.END_SCALE = 5e-2
_C.CLASSIFIER.LOSS_FUNC = "bce_logit"
_C.CLASSIFIER.T0 = 10
_C.CLASSIFIER.T1 = 10
_C.CLASSIFIER.T_MULT = 2
_C.CLASSIFIER.N_T = 0.5
_C.CLASSIFIER.GAMMA = 0.97
_C.CLASSIFIER.EPOCH_NUM = 120
_C.CLASSIFIER.ACCUMULATION_STEPS = 1
_C.CLASSIFIER.EVALUATION_STEPS = 5
_C.CLASSIFIER.PRINT_STEPS = 20

_C.CLASSIFIER.TEMPERATURE = 1

_C.CLASSIFIER.WEIGHT_DECAY = 0
_C.CLASSIFIER.L1_ALPHA = 0
_C.CLASSIFIER.L2_ALPHA = 0

_C.CLASSIFIER.FINETUNE_MODEL = False
_C.CLASSIFIER.FINETUNE = CfgNode()
_C.CLASSIFIER.FINETUNE.CKP = False
_C.CLASSIFIER.FINETUNE.BASE_LR = 6e-6
_C.CLASSIFIER.FINETUNE.END_LR = 1e-8
_C.CLASSIFIER.FINETUNE.LOSS_FUNC = "bce_logit"
_C.CLASSIFIER.FINETUNE.T0 = 10
_C.CLASSIFIER.FINETUNE.N_T = 0.5
_C.CLASSIFIER.FINETUNE.EPOCH_NUM = 0
_C.CLASSIFIER.FINETUNE.ACCUMULATION_STEPS = 3
_C.CLASSIFIER.FINETUNE.EVALUATION_STEPS = 5
_C.CLASSIFIER.FINETUNE.PRINT_STEPS = 20
_C.CLASSIFIER.FINETUNE.WEIGHT_DECAY = 2
_C.CLASSIFIER.FINETUNE.L1_ALPHA = 0
_C.CLASSIFIER.FINETUNE.L2_ALPHA = 0

_C.CLASSIFIER.CLASSES_NUM = 10
_C.CLASSIFIER.LOCATIONS = ['cytoplasm', 'cytoskeleton', 'endoplasmic reticulum', 'golgi apparatus', 'lysosomes', 'mitochondria',
                'nucleoli', 'nucleus', 'plasma membrane', 'vesicles']

_C.CLASSIFIER.NECK_DIM = 512
_C.CLASSIFIER.DROP_RATE = 0
_C.CLASSIFIER.ATTN_DROP_RATE = 0
_C.CLASSIFIER.DROP_PATH_RATE = 0
_C.CLASSIFIER.HEAD_DROP_RATE = 0.1


# ---------------------------------------------------------------------------- #
# Train options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

_C.TRAIN.BATCH_SIZE = 12
_C.TRAIN.EVAL_PERIOD = 20
_C.TRAIN.MIXED_PRECISION = True


# ---------------------------------------------------------------------------- #
# Test options.
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

_C.TEST.BATCH_SIZE = 12


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

_C.DATA_LOADER.NUM_WORKERS = 6
_C.DATA_LOADER.PIN_MEMORY = True


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()


dir_prefixs = {"GraphLoc": "data/GraphLoc/GraphLoc",
                "MSTLoc": "data/MSTLoc/MSTLoc",
                "laceDNN": "data/laceDNN/laceDNN",
                "IHC": "data/data",
                "cancer": "data/cancer/"}


labelLists = {"GraphLoc": ['cytoplasm', 'endoplasmic reticulum', 'golgi apparatus', 'mitochondria', 'nucleus', 'vesicles'],
            "MSTLoc": ['cytoplasm', 'endoplasmic reticulum', 'golgi apparatus', 'mitochondria', 'nucleus', 'vesicles'],
            "laceDNN": ['cytoplasm', 'golgi apparatus', 'mitochondria', 'nucleus', 'plasma membrane'],
            "IHC": ['cytoplasm', 'endoplasmic reticulum', 'mitochondria', 'nucleus', 'plasma membrane']}
