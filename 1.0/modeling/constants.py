from enum import Enum


class Constants(Enum):
    # checkpoint details
    CHECKPOINT_NAME = "torch_model.pt"

    # modeling constants
    EPS = 1e-8
    CONTEXT_LEN_FINE = 512
    CONTEXT_LEN_COARSE = 512
    AGG_FACTOR = 60    
