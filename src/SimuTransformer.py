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
        keys = key.split(".")
        value = self.ruleset
        for k in keys:
            if k in value:
                value = value[k]
            else:
                raise Errors.RulesetKeyError(key)
        return value

    def load_image(self, source):
        return cv2.imread(os.path.join(self.get_value("input.directory"), source))

    def save_image(self, image, save_as):
        cv2.imwrite(os.path.join(self.get_value("output.directory"), save_as), image)

    def apply(self, image, affine):
        size = self.get_value("input.size")
        r = self.get_value("output.size") / size

        # 平行移動
        affine[0][2] *= size
        affine[1][2] *= size

        # 出力サイズに応じたリサイズ
        affine *= r
        output_size = (int(size * r), int(size * r))

        flags = self.get_value("options.interpolation_flags")

        self.dump(affine=affine, output_size=output_size, flags=flags)
        return cv2.warpAffine(image, affine, output_size, flags)

    def transform(self):
        for rule in self.ruleset["rules"]:
            self.dump("--------------------")
            self.dump(rule=rule)

            image = self.load_image(rule["source"])
            if self.get_value("options.transparent"):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

            affine = tf.nothing()
            for convert in rule["converts"]:
                if hasattr(tf, convert):
                    method = getattr(tf, convert)
                    affine += method()

                else:
                    raise Errors.ConvertKeyError(convert)

            image = self.apply(image, affine)

            self.save_image(image, rule["save_as"])
