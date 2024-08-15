import argparse

from src import SimuTransformer
from src import Ruleset

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
        ruleset = Ruleset.Ruleset(args.file_name)
        simu_tranformer = SimuTransformer.SimuTransformer(ruleset, args.debug)
        simu_tranformer.transform()

    except Exception as e:
        if args.debug:
            raise e
        print(f"エラーが発生しました: {e}")
