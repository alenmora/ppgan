# PGGAN sample

## 初期設定
訓練前に実行してください。

1. モジュールのインストール
特に必要なし

## 訓練手順
### 1. データの準備
data フォルダ以下に画像データを格納する．現在利用できるものとしては，
/home/2019A/data/dentsu/images
に格納した二つのデータセットがある．

``` shell
-rw-r--r-- 1 taichi taichi 292179167 Dec  5 06:21 anime_face_dentsu.zip
-rw-r--r-- 1 taichi taichi 462126466 Dec  5 06:24 anime-faces.zip
```
anime_face_dentsuは電通側から提供されたもの，anime-facesはkaggleからダウンロードしたものである．

### 2. config ファイルの設定
config.py を適宜編集する。

#### 2-a. 最初から訓練する場合：
config.py:
- startRes=4
- startStage='stab'
を指定する

modelCheckpoint_<n>_[fade|stab]_37500.pth.tar
のような形式のモデルのチェックポイントファイルが生成されます。

訓練の順番は以下のステージを踏みます。
4*4 stable -> 8*8 fade in -> 8*8 stable -> 16*16 fade in ->16*16 stable

#### 2-b.　途中から訓練を再開する場合：
訓練の順番にしたがって、config.pyのなかの変数を指定すること。

再開する際に、モデルのチェックポイントファイルと同時に startRes,startStaege を指定してください。

例①：
modelFname='modelCheckpoint_8_stab_37500.pth.tar'　を利用する場合：
- startRes=16
- startStage='fade'
を指定する

例②：
modelFname='modelCheckpoint_8_fade_37500.pth.tar' を利用する場合：
- startRes=8
- startStage='stab'
を指定する

### 3. 以下のコマンドを実行する
 $ python train.py


## 生成手順

### 1. configファイルの設定
config.pyを編集してモデルのスナップショットファイル他を指定する

指定する項目は以下のようになる。
- config.gen_image_path
- config.LOG_DIR　　 : ログ出力場所
- config.logDir     : LOG_DIR　の直下に作成する各実験ごとのログの保存フォルダ名
- config.modelFname : モデルのチェックポイントファイルのパス


### 2. 以下のコマンドを実行する
(実行するたびに画像1枚生成され指定のpathに保存する)
$ python generator.py


## 画像の特徴ベクトルを抽出する手順
feature_extractor.pyを実行する。

引数'--img_path' にgeneratorで生成した画像のパス（ファイル名も含む）を指定する。
相対パスを指定した場合は、feature_extractor.py からの相対パスが指定されたことになる。

例：
$ python feature_extractor.py --img_path image_generated/test.jpg

実行すると画像の特徴値として1000次元のベクトルが生成される。
