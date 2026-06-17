from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGE_DIR = RAW_DIR / "images"
AUDIO_DIR = RAW_DIR / "audios"
METADATA_PATH = RAW_DIR / "metadata.csv"

PROCESSED_DIR = DATA_DIR / "processed"
SPECTROGRAM_DIR = PROCESSED_DIR / "spectrograms"

SPLIT_DIR = DATA_DIR / "splits"
TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV = SPLIT_DIR / "val.csv"
TEST_CSV = SPLIT_DIR / "test.csv"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

LABEL_MAP = {
    "unripe": 0,
    "ripe": 1,
    "overripe": 2,
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

NUM_CLASSES = len(LABEL_MAP)
IMAGE_SIZE = 224
SAMPLE_RATE = 22050
AUDIO_DURATION = 2.0
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
