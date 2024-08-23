import argparse
import glob
import os

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

    files = glob.glob(args.file_name)

    # 一致したファイルを1つずつ開く
    for file in files:
        try:
            _, ext = os.path.splitext(file)
            if ext == ".yml":
                ruleset = Ruleset.Ruleset(file)
                simu_tranformer = SimuTransformer.SimuTransformer(ruleset, args.debug)
                simu_tranformer.transform()

        except Exception as e:
            if args.debug:
                raise e
            print(f"エラーが発生しました: {e}")
