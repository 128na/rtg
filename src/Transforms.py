import numpy as np
import cv2

SGL = 0.25
DBL = SGL * 2
HLF = SGL / 2


class ImageManipulation:
    @staticmethod
    def empty_image(wh: tuple[int, int]) -> cv2.typing.MatLike:
        (w, h) = wh
        return np.zeros((h, w, 4), dtype=np.uint8)

    @staticmethod
    def empty_affine() -> cv2.typing.MatLike:
        return np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
            ],
            np.float32,
        )

    @staticmethod
    def crop(
        image: cv2.typing.MatLike, xy: tuple[int, int], size: int
    ) -> cv2.typing.MatLike:
        (x, y) = xy
        return image[y : y + size, x : x + size]

    @staticmethod
    def load_image(path: str):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        return image

    @staticmethod
    def save_image(image: cv2.typing.MatLike, path: str):
        cv2.imwrite(path, image)

    @staticmethod
    def apply(image: cv2.typing.MatLike, affine: cv2.typing.MatLike, size: int, r: int):
        h, w = image.shape[:2]

        # 平行移動
        affine[0][2] *= w
        affine[1][2] *= h

        # 出力サイズに応じたリサイズ
        affine *= size / h * r
        output_size = (int(size * r), int(size * r))

        return cv2.warpAffine(image, affine, output_size)

    @staticmethod
    def expand(image: cv2.typing.MatLike, r: int):
        h, w = image.shape[:2]
        output_size = (int(w * r), int(h * r))
        return cv2.resize(image, output_size)

    @staticmethod
    def shrink(image: cv2.typing.MatLike, r: int, f: int):
        h, w = image.shape[:2]
        output_size = (int(w / r), int(h / r))
        return cv2.resize(image, output_size, interpolation=f)

    @staticmethod
    def paste(
        base: cv2.typing.MatLike, overlay: cv2.typing.MatLike, xy: tuple[int, int]
    ):
        (x, y) = xy
        h, w = overlay.shape[:2]
        base[y : y + h, x : x + w] = overlay[:h, :w]
        return base


class Transforms:
    """
    線形変換行列定義クラス
    """

    @staticmethod
    def keep():
        """
        そのまま出力(正則行列)。resolutionによる拡大・縮小の影響は受ける。
        """
        return np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
            ],
            np.float32,
        )

    @staticmethod
    def resize(r):
        """
        リサイズ
        """
        return np.array(
            [
                [r, 0, 0],
                [0, r, 0],
            ],
            np.float32,
        )

    @staticmethod
    def reverse():
        """
        タイル画像からテクスチャ画像を逆生成する
        """
        return np.array(
            [
                [2, 4, -1.5],
                [-2, 4, -0.5],
            ],
            np.float32,
        )

    @staticmethod
    def to_n():
        """
        テクスチャの上をタイル画像の北へ向けて変換する
        """
        return np.array(
            [
                [DBL, -DBL, DBL],
                [SGL, SGL, DBL],
            ],
            np.float32,
        )

    @staticmethod
    def to_w():
        """
        テクスチャの上をタイル画像の西へ向けて変換する
        """
        return np.array(
            [
                [DBL, DBL, (DBL - DBL)],
                [-SGL, SGL, (DBL + SGL)],
            ],
            np.float32,
        )

    @staticmethod
    def to_e():
        """
        テクスチャの上をタイル画像の東へ向けて変換する
        """
        return np.array(
            [
                [-DBL, -DBL, (DBL + DBL)],
                [SGL, -SGL, (DBL + SGL)],
            ],
            np.float32,
        )

    @staticmethod
    def to_s():
        """
        テクスチャの上をタイル画像の南へ向けて変換する
        """
        return np.array(
            [
                [-DBL, DBL, DBL],
                [-SGL, -SGL, (DBL + DBL)],
            ],
            np.float32,
        )

    @staticmethod
    def to_up2():
        """
        テクスチャの右を緩坂2段分上げる
        """
        return np.array(
            [
                [0, 0, 0],
                [0, SGL, -SGL],
            ],
            np.float32,
        )

    @staticmethod
    def to_up():
        """
        テクスチャの右を緩坂1段分上げる
        """
        return np.array(
            [
                [0, 0, 0],
                [0, HLF, -HLF],
            ],
            np.float32,
        )
