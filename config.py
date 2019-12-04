# PGGAN 設定パラメータ

# logを保存するfolder
LOG_DIR='./log/'

# 訓練再start時に使用する
# 再開したい実験のlogフォルダー
# 新規実験を開始するたびに新しいlog用のフォルダーを切ることをおすすめ。
# よって、以下のようなディレクトリの構造が望ましい。
# ------------------------------------
# --LOG_DIR('./log/')
#   |
#    --logDir('experiment1/')
#   |
#    --logDir('experiment2/')
#   |
#    --logDir('experiment3/')
# ......
# ------------------------------------
logDir='experiment1/'

# 学習データが保存されているフォルダー
DATA_PATH='./data/anime/'

# 訓練再start 時に使用する
# 再開するsnapshotを指定する
# 初期状態は''
modelFname='modelCheckpoint_32_stab_37500.pth.tar'

# スナップショットを保存するかどうか
saveModel=True

## Hyperparameters
# Discriminator Learning Rate
dLR=1e-3

# Generator Learning Rate
gLR=1e-3

# Size of noise vector
latentSize=512

# Batch sizes for each resolution
# 4*4の時に16 size, 8*8の時に16 ...
batchSizes={4:16, 8:16, 16:16, 32:16, 64:8, 128:4, 256:4}

# Dict of Resolutions  
resolutions=[4, 8, 16, 32, 64, 128, 256]

# Other
# real of samples for each stage
samplesPerStage=600000

# log every x steps
logStep=2000

# 訓練再start 時に使用する
# 解像度を指定する、初期状態では4
#startRes=32
startRes=4

# 訓練再start 時に使用する
# 4*4 stable -> 8*8 fade in -> 8*8 stable -> 16*16 fade in ->16*16 stable
# のようにステップが分かれて学習が行われる。
# 再開時にどのステップから再開するかを指定する　
# stable -> stab, fade in->fade
#startStage='fade'
startStage='stab'

# 画像を生成する際の画像の出力先
# eg : './image_generated/test.jpg'
# 'test.jpg'とあるように、画像のファイル名とファイル形式まで指定すること。
gen_image_path=''

