from src import Ruleset
from src.Transforms import ImageManipulation as im
from src import Errors


class SimuTransformer:
    def __init__(self, ruleset: Ruleset.Ruleset, debug=False):
        self.ruleset = ruleset
        self.debug = debug
        self.dump(ruleset=ruleset.__dict__)

    def dump(self, msg=None, *args, **kwargs):
        if self.debug:
            if msg:
                print(msg)
            if args:
                print(args)
            if kwargs:
                print(kwargs)

    def transform(self):
        for fi, file in enumerate(self.ruleset.files):
            self.dump("--------------------")
            self.dump(file.__dict__)

            original = im.load_image(self.ruleset.resolve_path(file.source_path()))
            result = im.empty_image(file.dest_size())

            for ri, rule in enumerate(file.rules):
                self.dump(rule=rule.__dict__)

                source_size = rule.source_size() or file.source_default_size()
                if source_size is None:
                    raise Errors.MissingParamError(f"files.{fi}.rules.{ri}.source.size")

                image = im.crop(
                    original, rule.source_location(source_size), source_size
                )

                resolution = self.ruleset.resolution()
                if resolution != 1:
                    image = im.expand(image, resolution)

                affine = im.empty_affine()
                for ci, convert in enumerate(rule.converts):
                    affine = convert.apply(affine)

                dest_size = rule.dest_size() or file.dest_default_size()
                if dest_size is None:
                    raise Errors.MissingParamError(f"files.{fi}.rules.{ri}.dest.size")

                image = im.apply(image, affine, dest_size, resolution)
                if resolution != 1:
                    image = im.shrink(
                        image, resolution, self.ruleset.interpolation_flags()
                    )

                result = im.paste(result, image, rule.dest_location(dest_size))

            # im.dump(image)
            im.save_image(result, self.ruleset.resolve_path(file.dest_path()))
