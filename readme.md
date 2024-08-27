# Road Tile Generator

テクスチャ画像から線形変換を利用してSimutransの道路アドオン用タイル画像を生成するツールです。

![テクスチャ画像からの変換イメージ](./refs/thumb.png)

# 導入 / Installation

事前にpython（3.12以上）のインストールが必要です。

```
pip install -r requirements.txt
```

# 使い方 / Usage

```
# 指定yamlファイルの変換を実行
python .\main.py .\demo\demo.yml
# 指定ディレクトリ内にあるすべてのyamlファイルの変換を実行
python .\main.py .\demo\
```
変換の都合上、タイル外周部分はアンチエイリアスがかかり透過色との相性が悪いです。
使用するpakサイズよりも一回り大きめで出力し、別ツール([simutrans-image-merger](https://github.com/128na/simutrans-image-merger)など)の併用をお勧めします。

1. テクスチャを512px四方で作成
2. このツールで変形して256pxの画像を生成
3. 別ツールで128px四方にトリミング、特殊色削除
4. makeobjでpak作成

## 用語

- テクスチャ画像
    線形変換前の正方形の画像のこと
- タイル画像
    Simutrans用の角度に線形変換された画像のこと


## パラメーター

設定例は [demo.yml](./demo/demo.yml) を確認してください。

```yml
title: "demo"
# [任意] この定義全体の名前。--debug有効時に表示されます
description: "demo"
# [任意] この定義全体の説明。--debug有効時に表示されます
rtg_version: 1
# [必須] このアプリのメジャーバージョンです。バージョンが変わるとyamlファイルの互換性がなくなります。
options:
  interpolation_flags: 3    
  # [必須] 変換時の補完モード。後述
  resolution: 2             
  # [必須] 線形変換時の拡大倍率。後述
before_apply:               
# [任意] 入力画像に対してrules適用の前に実行する処理を指定します。後述
after_apply:                
# [任意] 出力画像に対してrules適用の後に実行する処理を指定します。後述
files:
  - name: "sample1"             
  # [必須] ファイルごとの名前。--debug有効時に表示されます
    source:
      path: "path/to/in.png"    
      # [必須] 入力画像のこのymlファイルを起点とした相対パス
      default_size: 512         
      # [任意] 入力画像のグリッドサイズデフォルト値(px)
    dest:
      path: "path/to/out.png"   
      # [必須] 出力画像のこのymlファイルを起点とした相対パス
      width: 1024               
      # [必須] 出力画像の幅(px)
      height: 512               
      # [必須] 出力画像の高さ(px)
      default_size: 256         
      # [任意] 出力画像のグリッドサイズデフォルト値(px)
    rules:
      - name: s                 
      # [必須] 変換ルールごとの名前。--debug有効時に表示されます
        source:                 
          location: "0.1"       
          # [必須] 入力画像のどの範囲を変換するかを指定します。datでの指定と同じです。
          size: 256             
          # [任意] 入力画像のグリッドサイズを指定します(px)未指定の場合デフォルト値が使用されます。
          offset:               
          # [任意] 入力画像のグリッド指定時のオフセット座標を指定できます
            x: 0                
            y: 0                
        dest:                   
          location: "1.0"       
          # [必須] 変換した画像を出力画像のどの場所に貼り付けるかを指定します。datでの指定と同じです。
          size: 256             
          # [任意] 出力画像のグリッドサイズを指定します(px)未指定の場合デフォルト値が使用されます。
          offset:               
          # [任意] 出力画像のグリッド指定時のオフセット座標を指定できます
            x: 0                
            y: 0                
        converts:               
        # [必須] 変換処理を指定します。後述
```


### `options.interpolation_flags`

変換時の補完モード。cv::InterpolationFlagsで定義されている値を指定できます。
https://docs.opencv.org/4.10.0/da/d54/group__imgproc__transform.html

### `options.resolution`

1以上を設定すると線形変換前に画像を指定倍率拡大、返還後に縮小して処理します。
アンチエイリアスの効き方が変わるので好みに合わせて設定してください。

resolution=1
![resolution=1での変換イメージ](./refs/r1.png)
resolution=2
![resolution=2での変換イメージ](./refs/r2.png)
resolution=4
![resolution=4での変換イメージ](./refs/r4.png)

### `before_apply`, `after_apply`

画像の編集操作を指定します。
使用可能な編集操作は [ImageEditクラス](./src/Transforms.py) を確認してください。

### `*.location`

画像の位置を指定します。datでの指定と同じです。

例） size=64, location=1,2の場合、(y,x) = (64,128)を始点に(w,h) = (64, 64)の領域となります

### `*.converts`

画像の線形変換方法を指定します。
使用可能な変換は [Transformsクラス](./src/Transforms.py) を確認してください。
