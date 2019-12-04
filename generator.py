import torch
from torch import FloatTensor as FT
from models import Generator
import dataUtils
import config


class Generator_for_eval:
    """
    image generator
    """
    def __init__(self):
        # Paths
        self.LOG_DIR = config.LOG_DIR
        self.logDir = config.logDir
        self.modelFname = config.modelFname

        # size of input noise (default:512)
        self.noise_size = config.latentSize

        self.loadPretrainedWts()

    def loadPretrainedWts(self):
        """
        create generator & load trained weights
        """
        self.logDir = self.LOG_DIR + self.logDir
        self.gen = Generator().cuda()

        wtsDict = torch.load(self.logDir,
                             map_location=lambda storage, loc: storage)
        self.gen.load_state_dict(wtsDict['gen'])

    def getNoise(self):
        return FT(self.noise_size).normal_().cuda()


if __name__ == "__main__":
    eval = Generator_for_eval()
    generator = eval.gen
    noise = eval.getNoise()

    # generatorに渡す二番目の引数で生成する画像のsizeを指定する
    # 対照関係は以下参照
    # 0:pixel=2*2, 1:pixel=4*4, 2:pixel=8*8, 3:pixel=16*16 4:pixel=32*32...
    output_tensor = generator(noise, 4)
    output_image = dataUtils.tensorToImage(output_tensor[0])

    print('image size : ', str(output_tensor.shape[2]) + '*' + str(output_tensor.shape[2]))
    output_image.save(config.gen_image_path)