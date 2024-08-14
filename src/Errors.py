class RTGVersionError(ValueError):
    def __init__(self, support_version):
        super().__init__(f"使用できるrtg_versionは { support_version } のみです")


class RulesetKeyError(ValueError):
    def __init__(self, key):
        super().__init__(f"'{ key }'の取得に失敗しました")


class ConvertKeyError(ValueError):
    def __init__(self, key):
        super().__init__(f"'{key}'の変換は存在しません。")
