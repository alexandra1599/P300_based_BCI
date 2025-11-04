# Configuration file

# UDP settings

UDP_MARKER = {"IP": "127.0.0.1", "PORT": 12345}

# EEG Settings
CAP_TYPE = 32
LOWCUT = 1  # Hz
HIGHCUT = 40  # Hz
FS = 512  # Sampling frequency in Hz

EEG_CHANNEL_NAMES = ["ALL"]
EOG_CHANNEL_NAMES = ["AUX1"]
EOG_TOGGLE = 0  # Toggle to enable or disbale EOG processing (1 = enabled, 0 = disabled)

# Experiment Parameters
TOTAL_TRIALS = 50  # Total number of trials
N_SPLITS = 5  # Number of splits for KFold Cross Validation
TIMING = True
SHAPE_MAX = 0.7  # max fill
SAPE_MIN = 0.5  # min fill

# Classification Parameters
CLASSIFY_WINDOW = 800  # Duration of EEG data window for classification in ms
ACCURACY_THRESHOLD = 0.55  # Accuracy threshold to determine "Correct"
THRESHOLD_TARGET = 0.5  # Threshold for target "Correct" : P300 detected
THRESHOLD_NONTARGET = 0.5  # Threshold for non-target "Correct": P300 not detected
RELAXATION_RATIO = 0.5
MIN_PREDICTIONS = (
    10  # Min number of prediction during online experiment vefore decoder can end early
)
STEP_SIZE = 1 / 16
CLASSIFICATION_OFFSET = 0  # Offset for "classification window" starting index
CLASSIFICATION_SCHEME_OPT = "TIMESERIES"
CAR_TOGGLE = 0  # Apply common average referencing during online
SELECT_CHANNELS = 0  # toggle to select specific channels or not
INTEGRATOR_ALPHA = (
    0.95  # Defines how fast accumulated probability may change as new data comes in
)
SHRINKAGE_PARAM = 0.1  # Hyperparameter for shrinkage regularization

# Screen Dimensions
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

FILTER_BUFFER_SIZE = 2048  # 4s at 512 Hz

P300_CHANNEL_NAMES = ["Pz", "Fpz", "Fz", "P3", "P4", "POz"]
# P300_CHANNEL_NAMES = ["Pz", "Cpz", "Fz", "P1", "P2", "POz"]

# Relevant Directories
WORKING_DIR = "/home/alexandra-admin/Documents/PhD/Task Code/"
DATA_DIR = "/home/alexandra-admin/Documents/CurrentStudy"
OUTPUT_ONLINE_DIR = "/home/alexandra-admin/Documents/Online"

MODEL_PATH = "/home/alexandra-admin/Documents/saved_models/"
DATA_FILE_PATH = "/home/alexandra-admin/Documents/CurrentStudy/sub-P001/ses-S001N2/eeg/sub-P001_ses-S001N2_task-Default_run-001OFFLINE_eeg.xdf"

TRAINING_SUBJECT = "102"
TRAINING_SESSION = "001OFFLINE"

USE_PREVIOUS_ONLINE_STATS = False  # for z-score of data coming in - this defines starting point, False = use stats from training, True = use previous online stats

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
blue = (0, 0, 255)
red = (255, 0, 0)
green = (0, 255, 0)
GREEN = (0, 255, 0)

orange = (255, 165, 0)

# Software Triggers
TRIGGERS = {
    "TRIAL_START": "0",
    "BUTTON_PRESS_MATCH": "100",
    "BUTTON_PRESS_NON_MATCH": "200",
    "TIMEOUT": "300",
    "TRIAL_END": "400",
    "MATCH": "11",
    "NON_MATCH": "12",
    "CORRECT": "1",
    "INCORRECT": "2",
    "TRIAL_PROBS": "3000",
    "TARGET_PROBS": "5000",
    "NON_TARGET_PROBS": "6000",
    "Correct Prediction": "33",
    "Incorrect Prediction": "44",
    "Prediction_end": "4000",
}


# Time Constants
fixation_time = 0.8
blink_time = 1.5
