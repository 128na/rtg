import yaml
from src import Errors
import os
import cv2
from src.Transforms import Transforms as tf
from src.Transforms import ImageEdit as ie

RTG_VERSION = 1


class Convert:
    name: str
    args: list | None

    def __init__(self, convert):
        if isinstance(convert, str):
            self.name = convert
            self.args = None
        else:
            self.name, *self.args = convert

    def apply(self, affine: cv2.typing.MatLike) -> cv2.typing.MatLike:
        if hasattr(tf, self.name):
            method = getattr(tf, self.name)
            if self.args:
                return affine + method(self.args)
            return affine + method()

        else:
            raise Errors.ConvertKeyError(self.name)


class Rule:
    name: str
    source: dict
    dest: dict
    converts: list[Convert]

    def __init__(self, rule):
        self.name = rule.get("name", "")
        self.source = rule["source"]
        self.dest = rule["dest"]
        self.converts = list(map(lambda c: Convert(c), rule["converts"]))

    def source_size(self) -> int | None:
        if "size" in self.source:
            return self.source["size"]

    def dest_size(self) -> int | None:
        if "size" in self.dest:
            return self.dest["size"]

    def source_location(self, size: int) -> tuple[int, int]:
        (y, x) = self.source["location"].split(".")

        return (int(x) * size, int(y) * size)

    def dest_location(self, size: int) -> tuple[int, int]:
        (y, x) = self.dest["location"].split(".")

        return (int(x) * size, int(y) * size)

    def dest_offset(self) -> tuple[int, int]:
        return (
            self.dest.get("offset", {}).get("x", 0),
            self.dest.get("offset", {}).get("y", 0),
        )

    def source_offset(self) -> tuple[int, int]:
        return (
            self.source.get("offset", {}).get("x", 0),
            self.source.get("offset", {}).get("y", 0),
        )


class Edit:
    name: str
    args: list | None

    def __init__(self, edit):
        if isinstance(edit, str):
            self.name = edit
            self.args = None
        else:
            self.name, *self.args = edit

    def apply(
        self, image: cv2.typing.MatLike, ruleset: "Ruleset"
    ) -> cv2.typing.MatLike:
        if hasattr(ie, self.name):
            method = getattr(ie, self.name)
            if self.args:
                return method(image, ruleset, self.args)
            return method(image, ruleset)

        else:
            raise Errors.EditKeyError(self.name)


class File:
    name: str
    source: dict
    dest: dict
    rules: list[Rule]
    before_apply: list[Edit]
    after_apply: list[Edit]

    def __init__(self, file):
        self.name = file.get("name", "")
        self.source = file["source"]
        self.dest = file["dest"]
        self.rules = list(map(lambda r: Rule(r), file["rules"]))
        self.before_apply = list(map(lambda m: Edit(m), file.get("before_apply", [])))
        self.after_apply = list(map(lambda m: Edit(m), file.get("after_apply", [])))

    def dest_size(self) -> tuple[int, int]:
        return (self.dest["width"], self.dest["height"])

    def source_path(self) -> str:
        return self.source["path"]

    def dest_path(self) -> str:
        return self.dest["path"]

    def source_default_size(self) -> int | None:
        return self.source.get("default_size")

    def dest_default_size(self) -> int | None:
        return self.dest.get("default_size")


class Ruleset:
    yaml_path: str
    data: dict
    files: list[File]
    options: dict

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.load()

    def load(self) -> None:
        try:
            with open(self.yaml_path, "r", encoding="utf-8") as file:
                self.data = yaml.safe_load(file)

                if self.data["rtg_version"] != RTG_VERSION:
                    raise Errors.RTGVersionError(RTG_VERSION)

                self.files = list(map(lambda f: File(f), self.data["files"]))
                self.options = self.data["options"]

        except FileNotFoundError:
            raise FileNotFoundError(f"ファイルが見つかりません: {self.yaml_path}")

    def resolve_path(self, path: str) -> str:
        dir = os.path.dirname(os.path.abspath(self.yaml_path))
        return os.path.join(dir, path)

    def resolution(self) -> int:
        return self.options["resolution"]

    def interpolation_flags(self) -> int:
        return self.options["interpolation_flags"]
