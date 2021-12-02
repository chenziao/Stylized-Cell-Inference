import numpy as np
from typing import Optional


class LFPScaler(object):

    def __init__(self, lfp: np.ndarray, scale: Optional[float] = None) -> None:
        self.lfp = lfp
        self.scale = scale

    def build_image(self, scale: Optional[float] = None) -> np.ndarray:
        if scale is None:
            if self.scale is None:
                raise ValueError("No scale was defined!")
            return np.clip(self.lfp, -self.scale, self.scale)
        return np.clip(self.lfp, -scale, scale)

