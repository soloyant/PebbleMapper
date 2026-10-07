import os
import sys
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(ROOT_DIR)  # vendored mrcnn lives at the repo root
from mrcnn.config import Config


class clastsConfig(Config):
    """Mask R-CNN configuration for clast detection."""
    NAME = "clasts"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 1  # Background + clasts

    STEPS_PER_EPOCH = 100

    DETECTION_MIN_CONFIDENCE = 0

    USE_MINI_MASK = False

    VALIDATION_STEPS = 50
    OPTIMIZER = "SGD" # can be ADAM or SGD
    DETECTION_MAX_INSTANCES = 1000

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_SCALE = 1

    MEAN_PIXEL = np.array([111.3883, 110.9057, 106.6095])

    # Square anchor side lengths in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    RPN_NMS_THRESHOLD = 0.9

    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    TRAIN_ROIS_PER_IMAGE = 128

    MAX_GT_INSTANCES = 1000


