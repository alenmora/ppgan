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
    parser.add_argument('--wtsFile', type=str, default='./ppgan3/128x128_final.pth.tar')
    parser.add_argument('--outFile', type=str, default='./generatedImages.png')
    parser.add_argument('--resolution', nargs='?')

    args, _ = parser.parse_known_args()

    device = torch.device('cpu')

    wts = loadPretrainedWts(args.wtsFile)
    gen = Generator(config).to(device)
    n = args.nImages

    gen.load_state_dict(wts['gen']) 

    z = utils.getNoise(bs = n, latentSize = config.latentSize, device = device)

    finalRes = int(args.wtsFile.split('/')[-1].split('x')[0])

    resLevel = None
    
    if args.resolution:
        resolution = int(args.resolution)
        if finalRes > resolution:
            resLevel = int(np.log2(resolution)-2)

    fakes = gen(z, curResLevel = resLevel)

    print('single image size: ', str(fakes.shape[2]) + 'x' + str(fakes.shape[2]))
    print(f'number of images: {n}')
    print(f'saving image to: {args.outFile}')

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

    utils.saveImage(fakes, args.outFile, nrow=nrows, padding=5)