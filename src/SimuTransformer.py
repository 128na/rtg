import cv2
import numpy as np

SGL = 0.25
DBL = SGL * 2
HLF = SGL / 2


class SimuTransformer:

    def identifical(self):
        return np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
            ],
            np.float32,
        )

    def to_n(self, h, w):
        return np.array(
            [
                [DBL, -DBL, DBL * w],
                [SGL, SGL, DBL * h],
            ],
            np.float32,
        )

    def to_w(self, h, w):
        return np.array(
            [
                [DBL, DBL, (DBL - DBL) * w],
                [-SGL, SGL, (DBL + SGL) * h],
            ],
            np.float32,
        )

    def to_e(self, h, w):
        return np.array(
            [
                [-DBL, -DBL, (DBL + DBL) * w],
                [SGL, -SGL, (DBL + SGL) * h],
            ],
            np.float32,
        )

    def to_s(self, h, w):
        return np.array(
            [
                [-DBL, DBL, DBL * w],
                [-SGL, -SGL, (DBL + DBL) * h],
            ],
            np.float32,
        )

    def to_up2(self, h, w):
        return np.array(
            [
                [0, 0, 0 * w],
                [0, SGL, -SGL * h],
            ],
            np.float32,
        )

    def to_up(self, h, w):
        return np.array(
            [
                [0, 0, 0 * w],
                [0, HLF, -HLF * h],
            ],
            np.float32,
        )
