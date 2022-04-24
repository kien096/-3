BATCH_SIZE = 16
NUMBER_CLASSES = 62
INPUT_SHAPE_IMAGE = (1, 128, 224)
EPOCHS = 2000
AUGMENTATION_DATA = True
PATIENCE = 40
GPU_NUM = 0
NUM_WORKERS = 8

PATH_DATA = 'data_only_audio'
JSON_NAME = 'train_data.json'
CLASS_JSON = 'standart_class.json'
IMAGES_PATH = 'images'
MASKS_PATH = 'masks'

DATASETS = ['val']

LEARNING_RATE = 0.0001
BACKBONE = 'None'
WEIGHTS = True
WEIGHTS_SPEECH = 'save_models/model_best-4_mobilenetv3_small_100.pth'
OUTPUT_ACTIVATION = 'softmax'
MODEL_NAME = 'mobilenetv3_small_100'
# MODEL_NAME = 'tf_mobilenetv3_large_075'
SAVE_MODEL_EVERY_EPOCH = False

LOGS = 'logs'
SAVE_MODELS = 'save_models'
SAVE_STATE_CURRENT_MODEL = 'save_state'
SAVE_BEST_MODEL = 'best_model'

PATH_LAST_STATE_MODEL = None

OVERLAY_AUDIO = True
AUDIO_LEN = 4500
FREQUENCY_DISCREDITING = 16000
