import cv2
from src.Transforms import Transforms as tf
import os
from src import Errors

SGL = 0.25
DBL = SGL * 2
HLF = SGL / 2


class SimuTransformer:
    debug = False
    ruleset = {}

    cachedKV = {}

    def __init__(self, debug=False):
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

    def get_value(self, key):
        if key not in self.cachedKV:
            keys = key.split(".")
            value = self.ruleset
            for k in keys:
                if k in value:
                    value = value[k]
                else:
                    raise Errors.RulesetKeyError(key)

                self.cachedKV[key] = value
        return self.cachedKV[key]

    def load_image(self, source):
        filename = os.path.join(self.get_value("input.directory"), source)
        self.dump(load_image=filename)
        return cv2.imread(filename)

    def save_image(self, image, save_as):
        filename = os.path.join(self.get_value("output.directory"), save_as)
        self.dump(save_image=filename)
        cv2.imwrite(filename, image)

    def apply(self, image, affine):
        size = self.get_value("input.size") * self.get_value("options.resolution")
        r = self.get_value("output.size") * self.get_value("options.resolution") / size

        # 平行移動
        affine[0][2] *= size
        affine[1][2] *= size

        # 出力サイズに応じたリサイズ
        affine *= r
        output_size = (int(size * r), int(size * r))

        self.dump(affine=affine, output_size=output_size)
        return cv2.warpAffine(image, affine, output_size)

    def expand(self, image):
        r = self.get_value("options.resolution")
        h, w = image.shape[:2]
        output_size = (int(w * r), int(h * r))
        return cv2.resize(image, output_size)

    def shrink(self, image):
        r = self.get_value("options.resolution")
        h, w = image.shape[:2]
        output_size = (int(w / r), int(h / r))
        flags = self.get_value("options.interpolation_flags")
        return cv2.resize(image, output_size, interpolation=flags)

    def transform(self):
        for rule in self.ruleset["rules"]:
            self.dump("--------------------")
            self.dump(rule=rule)

            image = self.load_image(rule["source"])
            if self.get_value("options.transparent"):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

            resolution = self.get_value("options.resolution")
            if resolution != 1:
                image = self.expand(image)

            affine = tf.nothing()
            for convert in rule["converts"]:
                if hasattr(tf, convert):
                    method = getattr(tf, convert)
                    affine += method()

                else:
                    raise Errors.ConvertKeyError(convert)

            image = self.apply(image, affine)

            if resolution != 1:
                image = self.shrink(image)

            self.save_image(image, rule["save_as"])
