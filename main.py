import cv2
import json
import argparse

from src import SimuTransformer


def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(
            f"JSONファイルの読み込み中にエラーが発生しました: {file_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="指定されたJSONファイルを読み込むスクリプト"
    )
    parser.add_argument("file_name", help="読み込みたいJSONファイルのパス")
    args = parser.parse_args()

    simu_tranformer = SimuTransformer.SimuTransformer()

    try:
        rules = load_json(args.file_name)
        for rule in rules:
            print(rule)
            input_path = rule["input"]
            output_path = rule["output"]

            image = cv2.imread(input_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

            affine = simu_tranformer.identifical()
            for convert in rule["converts"]:

                if hasattr(simu_tranformer, convert):
                    method = getattr(simu_tranformer, convert)
                    h, w = image.shape[:2]
                    affine += method(h, w)

                else:
                    raise AttributeError(
                        f"'{type(simu_tranformer).__name__}' object has no method '{convert}'"
                    )

            image = cv2.warpAffine(image, affine, (w, h), flags=cv2.INTER_CUBIC)

            cv2.imwrite(output_path, image)

    except Exception as e:
        print(f"エラーが発生しました: {e}")
