import torch
from torch import FloatTensor as FT
from models import Generator
import utils
import os
from config import config
import argparse
import math
import numpy as np

def loadPretrainedWts(dir):
    """
    load trained weights
    """
        
    if os.path.isfile(dir):
        try:
            wtsDict = torch.load(dir, map_location=lambda storage, loc: storage)
            return wtsDict
        except:
            print(f'ERROR: The weights in {dir} could not be loaded')
    else:
        print(f'ERROR: The file {dir} does not exist')    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('PGGAN_GEN')
    parser.add_argument('--nImages', type=int, default=20)
    parser.add_argument('--latentSize', nargs='?', tpye=int)
    parser.add_argument('--nChannels', type=int, default=3)
    parser.add_argument('--wtsFile', type=str, default='./pretrainedModels/128x128_final_128.pth.tar')
    parser.add_argument('--outputFile', type=str, default='./generatedImages.png')
    parser.add_argument('--resolution', nargs='?', type=int)

    args, _ = parser.parse_known_args()

    endRes = int(args.resolution) if args.resolution else int(args.wtsFile.split('/')[-1].split('x')[0])

    config.latentSize = args.latentSize if args.latentSize else int(args.wtsFile.split('/')[-1].split('_')[-1])
    config.endRes = endRes
    config.nChannels = args.nChannels

    device = torch.device('cpu')

    wts = loadPretrainedWts(args.wtsFile)
    gen = Generator(config).to(device)
    n = args.nImages

    gen.load_state_dict(wts['gen']) 

    z = utils.getNoise(bs = n, latentSize = args.latentSize, device = device)

    fakes = gen(z)

    print('single image size: ', str(fakes.shape[2]) + 'x' + str(fakes.shape[2]))
    print(f'number of images: {n}')
    print(f'saving image to: {args.outputFile}')

    nrows = 1

    if math.sqrt(n) == int(math.sqrt(n)):
        nrows = math.sqrt(n)
    
    elif n > 5:
        i = int(math.sqrt(n))
        while i > 2:
            if (n % i) == 0:
                nrows = i
                break

            i = i-1

    utils.saveImage(fakes, args.outputFile, nrow=nrows, padding=5)
