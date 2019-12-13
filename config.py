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
EXP_DIR='experiment1/'

# 学習データが保存されているフォルダー
DATA_PATH='./data/anime/'

# スナップショットを保存するかどうか
saveModel=True

# Use pretrained weights if available. Together with this, the pretrained weights' file
# complete address and name should be given in the preWtsFile variable. Otherwise, this is ignored
usePreWts = True
preWtsFile = None

## Hyperparameters
# Critic Learning Rate
dLR=1e-3
# Group size for the standard deviation calculation in the last block of the critic. If None, it is set equal to 4.
# To deactivate, make it 0 or 1
stdDevGroup = 4
# lambda and obj are hyperparameters to perform gradient penalization on the critic. It is done by adding a term of 
# the form lamb*(grad**2 - obj)**2 to the loss function, where grad is the gradient with respect to the input values
lamb = 10
obj = 1
# epsilon is a small parameter to stop the critic output to explode. This is done by adding a term of the form 
# epsilon*(output)**2 to the loss function
epsilon = 0.001

# Generator Learning Rate
gLR=1e-3

# Batch sizes for each resolution
# 4*4の時に16 size, 8*8の時に16 ...
batchSizes={4:16, 8:16, 16:16, 32:16, 64:8, 128:4, 256:4}

# Dict of Resolutions  
resolutions=[4, 8, 16, 32, 64, 128, 256]

# Other
# Number of real images shown before increasing the resolution. Must be divisible by all batch sizes
samplesWhileStable = 600000

#Number of real images shown wile fading in new layers. Must be divisible by all batch sizes
samplesWhileFade = 600000

# log every x steps
logStep=2000

# 訓練再start 時に使用する
# 解像度を指定する、初期状態では4
#startRes=32
startRes = 4
endRes = 256

# Size of noise vector. If none, latentSize = endRes/2
latentSize=None

# Parameters to calculate the number of channels for each resolution block
# from the equation nchannels = min(fmapBase/(2**(nBlock*fmapDecay)), fmapMax)
# if None, fmapMax = endRes/2, fmapBase = 8*endRes and fmapDecay = 1.0
fmapBase = None
fmapMax = None
fmapDecay = None

# Start step for the training. If different than None, the training will
# assume there have been performed startStep training loops already
startStep = None

# 画像を生成する際の画像の出力先
# eg : './image_generated/test.jpg'
# 'test.jpg'とあるように、画像のファイル名とファイル形式まで指定すること。
gen_image_path=''

