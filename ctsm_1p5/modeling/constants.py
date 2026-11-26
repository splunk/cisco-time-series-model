"""Geometry and quantiles of the released CTSM 1.5 checkpoint."""

FINE_CONTEXT_LENGTH: int = 4096
COARSE_CONTEXT_LENGTH: int = 512
AGG_FACTOR: int = 60
PATCH_LENGTH: int = 32
OUTPUT_PATCH_LENGTH: int = 128
HIDDEN_SIZE: int = 1280
INTERMEDIATE_SIZE: int = 1280
NUM_LAYERS: int = 25
NUM_HEADS: int = 16
NUM_KV_HEADS: int = 16
HEAD_DIM: int = 80
QUANTILES: tuple[float, ...] = (
    0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99,
)
MEDIAN_QUANTILE_INDEX: int = QUANTILES.index(0.5)
PAD_VALUE: float = 1123581321.0
TOLERANCE: float = 1e-6
RMS_NORM_EPS: float = 1e-6
EPS: float = 1e-8
