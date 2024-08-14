import numpy as np

SGL = 0.25
DBL = SGL * 2
HLF = SGL / 2


class Transforms:

    @staticmethod
    def nothing():
        return np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
            ],
            np.float32,
        )

    @staticmethod
    def to_n():
        return np.array(
            [
                [DBL, -DBL, DBL],
                [SGL, SGL, DBL],
            ],
            np.float32,
        )

    @staticmethod
    def to_w():
        return np.array(
            [
                [DBL, DBL, (DBL - DBL)],
                [-SGL, SGL, (DBL + SGL)],
            ],
            np.float32,
        )

    @staticmethod
    def to_e():
        return np.array(
            [
                [-DBL, -DBL, (DBL + DBL)],
                [SGL, -SGL, (DBL + SGL)],
            ],
            np.float32,
        )

    @staticmethod
    def to_s():
        return np.array(
            [
                [-DBL, DBL, DBL],
                [-SGL, -SGL, (DBL + DBL)],
            ],
            np.float32,
        )

    @staticmethod
    def to_up2():
        return np.array(
            [
                [0, 0, 0],
                [0, SGL, -SGL],
            ],
            np.float32,
        )

    @staticmethod
    def to_up():
        return np.array(
            [
                [0, 0, 0],
                [0, HLF, -HLF],
            ],
            np.float32,
        )
