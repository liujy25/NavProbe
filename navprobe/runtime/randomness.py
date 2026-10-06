"""Initialize local random streams once per episode, before environment creation."""
from importlib.util import find_spec
import random

import cv2
import numpy as np


def seed_local_randomness(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # OpenCV accepts a signed 32-bit integer; preserve the seed's bit pattern.
    cv2.setRNGSeed(seed if seed < 2**31 else seed - 2**32)
    if find_spec("torch") is not None:
        import torch

        # Includes every CUDA device, also when CUDA is initialized later.
        torch.manual_seed(seed)
