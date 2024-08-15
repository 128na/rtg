import cv2
from src.Transforms import Transforms as tf
import os
from src import Errors
import matplotlib.pyplot as plt
import numpy as np

SGL = 0.25
DBL = SGL * 2
HLF = SGL / 2


def data_get(value, key):
    keys = key.split(".")
    for k in keys:
        if k in value:
            value = value[k]
        else:
            raise AttributeError(f"missing key '{key}' in {value}")
    return value


def data_get_or_default(value, key, default):
    keys = key.split(".")
    for k in keys:
        if k in value:
            value = value[k]
        else:
            return default
    return value


class SimuTransformer:
    yaml_path = None
    debug = False
    ruleset = {}

    cachedKV = {}

    def __init__(self, yaml_path, debug=False):
        self.yaml_path = yaml_path
        self.debug = debug

    def set_ruleset(self, ruleset):
        self.ruleset = ruleset

        self.dump(ruleset=ruleset)

    def dump(self, msg=None, *args, **kwargs):
        if self.debug:
            if msg:
                print(msg)
            if args:
                print(args)
            if kwargs:
                print(kwargs)

    def crop(self, image, xy, size):
        return image[xy[1] : xy[1] + size, xy[0] : xy[0] + size]

    def resolve_path(self, file_path):
        dir = os.path.dirname(os.path.abspath(self.yaml_path))
        return os.path.join(dir, file_path)

    def load_image(self, path):
        file_path = self.resolve_path(path)
        self.dump(load_image=file_path)
        return cv2.imread(file_path)

    def save_image(self, image, path):
        file_path = self.resolve_path(path)
        self.dump(save_image=file_path)
        cv2.imwrite(file_path, image)

    def apply(self, image, affine, size, r):
        h, w = image.shape[:2]

        # 平行移動
        affine[0][2] *= w
        affine[1][2] *= h

        # 出力サイズに応じたリサイズ
        affine *= size / h * r
        output_size = (int(size * r), int(size * r))

        self.dump(affine=affine, output_size=output_size)
        return cv2.warpAffine(image, affine, output_size)

    def expand(self, image):
        r = data_get(self.ruleset, "options.resolution")
        h, w = image.shape[:2]
        output_size = (int(w * r), int(h * r))
        return cv2.resize(image, output_size)

    def shrink(self, image):
        r = data_get(self.ruleset, "options.resolution")
        h, w = image.shape[:2]
        output_size = (int(w / r), int(h / r))
        flags = data_get(self.ruleset, "options.interpolation_flags")
        return cv2.resize(image, output_size, interpolation=flags)

    def resolve_location(self, location, size):
        (x, y) = location.split(".")
        return (int(x) * size, int(y) * size)

    def paste(self, base, overlay, xy):
        (x, y) = xy
        h, w = overlay.shape[:2]
        base[y : y + h, x : x + w] = overlay[:h, :w]

        return base

    def transform(self):
        for file in data_get(self.ruleset, "files"):
            self.dump("--------------------")

            image = self.load_image(data_get(file, "in.path"))
            out = np.zeros(
                (data_get(file, "out.height"), data_get(file, "out.width"), 4),
                dtype=np.uint8,
            )
            if data_get(self.ruleset, "options.transparent"):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

            for rule in data_get(file, "rules"):
                self.dump(rule=rule)
                size = data_get(rule, "in.size")
                location = self.resolve_location(data_get(rule, "in.location"), size)
                self.dump(location=location)

                im = self.crop(image, location, size)

                resolution = data_get(self.ruleset, "options.resolution")
                if resolution != 1:
                    im = self.expand(im)

                affine = tf.nothing()
                for convert in rule["converts"]:
                    if hasattr(tf, convert):
                        method = getattr(tf, convert)
                        affine += method()

                    else:
                        raise Errors.ConvertKeyError(convert)

                out_size = data_get(rule, "out.size")
                im = self.apply(im, affine, out_size, resolution)
                out_location = self.resolve_location(
                    data_get(rule, "out.location"), out_size
                )
                if resolution != 1:
                    im = self.shrink(im)

                self.dump(out_location=out_location)
                out = self.paste(out, im, out_location)

            self.save_image(out, data_get(file, "out.path"))
            # plt.imshow(out)
            # plt.show()
