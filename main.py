import argparse
import yaml

from src import SimuTransformer
from src import Errors

RTG_VERSION = 0


def load_ruleset(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

            if "rtg_version" not in data or data["rtg_version"] != RTG_VERSION:
                raise Errors.RTGVersionErrorRTG(RTG_VERSION)

            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="指定されたJSONファイルを読み込むスクリプト"
    )
    parser.add_argument("file_name", help="JSONファイルのパス")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグモード",
        default=False,
    )
    args = parser.parse_args()

    try:
        ruleset = load_ruleset(args.file_name)
        simu_tranformer = SimuTransformer.SimuTransformer(args.debug)
        simu_tranformer.set_ruleset(ruleset)
        simu_tranformer.transform()

    except Exception as e:
        if args.debug:
            raise e
        print(f"エラーが発生しました: {e}")
