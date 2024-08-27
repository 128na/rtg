import numpy as np
import cv2
import matplotlib.pyplot as plt
from src import Ruleset

SGL = 0.25
DBL = SGL * 2
HLF = SGL / 2


class ImageManipulation:
    @staticmethod
    def dump(image):
        plt.figure(facecolor="black")
        plt.grid(True, color="white", linewidth=0.5)
        plt.gca().set_xticks(np.arange(0, image.shape[1], 16))
        plt.gca().set_yticks(np.arange(0, image.shape[0], 16))
        plt.gca().tick_params(length=0)  # グリッドの目盛りを消す
        plt.gca().set_aspect("equal")  # 正方形グリッドを維持
        plt.imshow(image)
        plt.show()

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
        image: cv2.typing.MatLike, xy: tuple[int, int], size: int, dxdy: tuple[int, int]
    ) -> cv2.typing.MatLike:
        (x, y) = xy
        (dx, dy) = dxdy
        return image[y + dy : y + dy + size, x + dx : x + dx + size]

    @staticmethod
    def load_image(path: str):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
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
        base: cv2.typing.MatLike,
        overlay: cv2.typing.MatLike,
        xy: tuple[int, int],
        dxdy: tuple[int, int],
    ):
        (x, y) = xy
        (dx, dy) = dxdy

        # 背景画像の指定された場所を切り出す
        h, w = overlay.shape[:2]
        roi = base[y + dy : y + dy + h, x + dx : x + dx + w]

        # 背景画像と重ねる画像の各チャンネルを分離する
        overlay_rgb = overlay[:, :, :3]  # 重ねる画像のRGB
        overlay_alpha = overlay[:, :, 3]  # 重ねる画像のαチャンネル

        base_rgb = roi[:, :, :3]  # 背景画像の該当領域のRGB
        base_alpha = roi[:, :, 3]  # 背景画像の該当領域のαチャンネル

        # αチャンネルを0〜1の範囲に正規化
        alpha_overlay = overlay_alpha.astype(float) / 255.0
        alpha_base = base_alpha.astype(float) / 255.0

        # ベース画像が透明 (alpha_base == 0) の場合、overlay_rgbをそのまま使用する
        blended_rgb = np.where(
            alpha_base[..., None] == 0,
            overlay_rgb,
            alpha_overlay[..., None] * overlay_rgb
            + (1 - alpha_overlay[..., None]) * base_rgb,
        ).astype(np.uint8)

        # アルファチャンネルのブレンド
        blended_alpha = (alpha_overlay + alpha_base * (1 - alpha_overlay)) * 255
        blended_alpha = blended_alpha.astype(np.uint8)

        # 最終的な合成画像を作成
        result = cv2.merge(
            (
                blended_rgb[:, :, 0],
                blended_rgb[:, :, 1],
                blended_rgb[:, :, 2],
                blended_alpha,
            )
        )

        # 合成した部分を背景画像に戻す
        base[y + dy : y + dy + h, x + dx : x + dx + w] = result

        return base


class ImageEdit:
    def shift(image, ruleset: Ruleset, args: list):
        # 画像がアルファチャンネルを持っているか確認
        if image.shape[2] == 4:
            # BGRとアルファチャンネルを分離
            bgr = image[:, :, :3]  # BGRチャンネル
            alpha = image[:, :, 3]  # アルファチャンネル
        else:
            bgr = image
            alpha = None  # アルファチャンネルがない場合

        # BGRからHSVに変換
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # HSVチャンネルを分離
        h, s, v = cv2.split(hsv)

        hue_shift = args[0]

        # OpenCVのHチャンネルは0〜179の範囲なので、360に対応させる
        # Hチャンネルを0〜360に拡張して計算
        h = (h.astype(np.int32) * 2 + hue_shift) % 360

        # 360度の色相を再度0〜179の範囲にマッピング
        h = (h // 2).astype(np.uint8)

        # チャンネルを再結合
        shifted_hsv = cv2.merge([h, s, v])

        # HSVからBGRに変換
        shifted_bgr = cv2.cvtColor(shifted_hsv, cv2.COLOR_HSV2BGR)

        # アルファチャンネルがある場合、BGRと再結合
        if alpha is not None:
            # アルファチャンネルとBGRを結合して出力画像を作成
            return cv2.merge([shifted_bgr, alpha])
        else:
            return shifted_bgr

    def merge(image, ruleset: Ruleset, args: list):
        path = args[0]
        xy = (args[1] if len(args) > 1 else 0, args[2] if len(args) > 2 else 0)
        dxdy = (args[3] if len(args) > 3 else 0, args[4] if len(args) > 4 else 0)
        overlay = ImageManipulation.load_image(ruleset.resolve_path(path))

        return ImageManipulation.paste(image, overlay, xy, dxdy)


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
    def to_up(args: list[int] = []):
        """
        テクスチャの右を緩坂1段分上げる
        """
        height = args[0] if len(args) > 0 else 1
        return np.array(
            [
                [0, 0, 0],
                [0, HLF * height, -HLF * height],
            ],
            np.float32,
        )
