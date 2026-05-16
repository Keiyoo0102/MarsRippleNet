"""
Default configuration for MarsRippleNet experiments.
"""

from pathlib import Path

# ── Project Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "DATA" / "Label" / "Augmented_Final" / "64"
OUTPUT_DIR = PROJECT_ROOT / "checkpoints"


# ── Training Hyperparameters ──────────────────────────────────────────────────

class TrainConfig:
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Optimizer & Scheduler
    OPTIMIZER = "AdamW"
    SCHEDULER = "CosineAnnealingLR"
    WARMUP_EPOCHS = 5

    # Data loading
    NUM_WORKERS = 4
    PIN_MEMORY = True
    SHUFFLE_TRAIN = True
    DROP_LAST_TRAIN = False

    # Random seed
    SEED = 42


# ── Model Configurations ───────────────────────────────────────────────────────

class ModelConfig:
    # Input/Output
    IN_CHANNELS = 1       # Mars HiRISE images are single-channel grayscale
    NUM_CLASSES = 4       # Mask, Straight, Sinuous, Complex

    # Encoder
    ENCODER = "efficientnet-b0"
    ENCODER_WEIGHTS = None

    # Decoder channels (5 blocks: block0-bottleneck, block1, ..., block4)
    DECODER_CHANNELS = (256, 128, 64, 32, 16)

    # SAFM injection positions (decoder block indices, 0-indexed from top)
    # Block 2 = 16×16 spatial resolution
    # Block 3 = 8×8 spatial resolution
    SAFM_BLOCK_INDICES = (2, 3)

    # DWR parameters
    DWR_REDUCTION = 4
    DWR_DILATIONS = (1, 3, 5)

    # SPM parameters
    SPM_REDUCTION = 16

    # SAFM fusion channels (None = auto = in_channels // 4)
    SAFM_FUSION_CHANNELS = None


# ── Loss Function Config ───────────────────────────────────────────────────────

class LossConfig:
    # Loss type: "ce" | "dice" | "combined"
    LOSS_TYPE = "combined"
    CE_WEIGHT = 0.5
    DICE_WEIGHT = 0.5

    # Class weights (Mask, Straight, Sinuous, Complex)
    CLASS_WEIGHTS = [1.0, 1.5, 1.5, 1.2]


# ── Data Configuration ─────────────────────────────────────────────────────────

class DataConfig:
    # Available patch sizes: 32, 48, 64
    PATCH_SIZE = 64

    # Normalization
    NORMALIZE = True
    NORMALIZE_MEAN = 0.5
    NORMALIZE_STD = 0.5


# ── Class Definitions ─────────────────────────────────────────────────────────

CLASS_NAMES = {
    0: "Masked",
    1: "Straight",
    2: "Sinuous",
    3: "Complex",
}

CLASS_COLORS = {
    0: (128, 128, 128),  # Gray
    1: (255, 0, 0),      # Red
    2: (0, 255, 0),      # Green
    3: (0, 0, 255),      # Blue
}

EVAL_CLASSES = [1, 2, 3]  # Exclude Masked (class 0) from metrics


# ── Convenience Export ─────────────────────────────────────────────────────────

Config = TrainConfig
__all__ = [
    "PROJECT_ROOT", "DATA_ROOT", "OUTPUT_DIR",
    "TrainConfig", "ModelConfig", "LossConfig", "DataConfig",
    "CLASS_NAMES", "CLASS_COLORS", "EVAL_CLASSES",
]
