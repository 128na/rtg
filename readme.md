# Road Tile Generator

平面画像からSimutransの道路アドオン用画像を線形変換して生成するツールです。

![平面画像からの変換イメージ](./refs/thumb.png)

# 導入 / Installation

事前にpythonのインストールが必要です。

```
pip install -r requirements.txt
```

# 使い方 / Usage

```
python .\main.py .\demo\demo.yml
```
変換の都合上、タイル外周部分はアンチエイリアスがかかり透過色との相性が悪いです。
使用するpakサイズよりも一回り大きめで出力し、別ツール([simutrans-image-merger](https://github.com/128na/simutrans-image-merger)など)の併用をお勧めします。

1. テクスチャを512px四方で作成
2. このツールで変形して256pxの画像を生成
3. 別ツールで128px四方にトリミング、特殊色削除
4. makeobjでpak作成

## パラメーター

主な設定は [demo.yml](./demo/demo.yml) を確認してください。

### rtg_version

このアプリのメジャーバージョンです。バージョンが変わるとyamlファイルの互換性がなくなります。

### options.interpolation_flags
変換時の補完モード

### options.resolution

1以上を設定すると線形変換前に画像を指定倍率拡大、返還後に縮小して処理します。
アンチエイリアスの効き方が変わるので好みに合わせて設定してください。

resolution=1
![resolution=1での変換イメージ](./refs/r1.png)
resolution=2
![resolution=2での変換イメージ](./refs/r2.png)
resolution=4
![resolution=4での変換イメージ](./refs/r4.png)

### *.location

画像の位置を指定します。datでの指定と同じです。

例） size=64, location=1,2の場合、(y,x) = (64,128)を始点に(w,h) = (64, 64)の領域となります

### files.rules.source.size, files.rules.dest.size

各画像の入出力サイズです。
未指定のときはデフォルト値(`files.souce.default_size`, `files.dest.default_size`)が使用されます。

### *.converts

画像の線形変換方法を指定します
使用可能な変換は [Transformsクラス](./src/Transforms.py) を確認してください。
