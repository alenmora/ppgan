import torch
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
    # When sampling the latent vector during training, extreme values are less likely to appear, 
    # and hence the generator is not sufficiently trained in these regions. Hence, we limit the 
    # values of the latent vector to be inside (-psiCut, psiCut)
    parser.add_argument('--psiCut', type=float, default=1.5)        
    parser.add_argument('--latentSize', nargs='?', type=int)
    parser.add_argument('--nChannels', type=int, default=3)
    parser.add_argument('--wtsFile', type=str, default='./pretrainedModels/64x64_final_256.pth.tar')
    parser.add_argument('--outputFile', type=str, default='./generated.png')
    parser.add_argument('--resolution', nargs='?', type=int)
    parser.add_argument('--createInterpolGif', action='store_true')

    args, _ = parser.parse_known_args()

    endRes = int(args.resolution) if args.resolution else int(args.wtsFile.split('/')[-1].split('x')[0])

    latentSize = args.latentSize if args.latentSize else int(args.wtsFile.split('/')[-1].split('_')[-1].split('.')[0])
    
    config.endRes = endRes
    config.nChannels = args.nChannels
    config.latentSize = latentSize

    device = torch.device('cpu')

    cut = abs(args.psiCut)
    wts = loadPretrainedWts(args.wtsFile)
    gen = Generator(config).to(device)
    n = args.nImages
    out = args.outputFile

    if n <= 0: 
        n = 20

    gen.load_state_dict(wts['gen']) 

    z = torch.tensor([])

    if args.createInterpolGif:
        out = args.outputFile.split('/')[0]+'/'+args.outputFile.split('/')[-1].split('.')[0]+'.gif'
        zstart = utils.getNoise(bs = 1, latentSize = latentSize, device = device)
        zend = utils.getNoise(bs = 1, latentSize = latentSize, device = device)
        z = zstart.clone()
        for i in range(1,n+1):
            alpha = i/n
            interp = (1-alpha)*zstart+alpha*zend
            z = torch.cat([z,interp],dim=0)

    else:
        z = utils.getNoise(bs = n, latentSize = latentSize, device = device)
    
    if cut != 0:
        z = torch.fmod(z,cut)
    else:
        z = z.new(*z.shape).fill_(0)

    fakes = gen(z)

    print('single image size: ', str(fakes.shape[2]) + 'x' + str(fakes.shape[2]))
    print(f'number of images{" in gif" if args.createInterpolGif else ""}: {n}')
    print(f'saving {"gif" if args.createInterpolGif else "image"} to: {out}')

    if args.createInterpolGif:
        utils.saveGif(fakes, out)

    else:
        nrows = 1

        if math.sqrt(n) == int(math.sqrt(n)):
            nrows = int(math.sqrt(n))
    
        elif n > 5:
            i = int(math.sqrt(n))
            while i > 2:
                if (n % i) == 0:
                    nrows = i
                    break

                i = i-1

        utils.saveImage(fakes, out, nrow=nrows, padding=5)
