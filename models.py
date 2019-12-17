import torch
import modelUtils, torch.nn as nn, numpy as np, torch.autograd as autograd

##############################################################
# Generator
##############################################################

def genConvBlock(net, inCh, outCh, kernelSize, stride=1, negSlope=0.2, padding=None):
    """
    This funtion appends and returns LIST of conv blocks for gen
    net (list): list to append the current block
    inCh (int): the number of channels in the input 
    outCh (int): the number of channels in the output
    kernelSize (int,list): the dimension(s) of the convolution kernel
    stride (int,list): the steps taken in each dimension
    negSlope (int): the slope for the negative values of the input in the leakyReLU
    padding (int, list): padding on each dimension. If none, padding is made to keep the size of the input tensor
    """
    assert kernelSize >= 1 and kernelSize % 2 == 1
    if padding == None: padding = int((kernelSize-1)/2)  #Make sure the output tensors for each channel are also 4x4 by adjusting the padding 
    net += [modelUtils.WSConv2d(inCh=inCh, outCh=outCh, kernelSize=kernelSize, stride=stride, padding=padding)]    
    net += [nn.LeakyReLU(negative_slope=negSlope)]
    net += [modelUtils.PixelNormalization()]
    return net


def toRGBBlock(inCh, outCh, kernelSize=1, stride=1):
    """
    This returns a list containing a post processing block (channel to RGB or similar)
    inCh (int): the number of channels in the input 
    outCh (int): the number of color channels
    kernelSize (int,list): the dimension(s) of the convolution kernel
    stride (int,list): the steps taken in each dimension
    """
    assert kernelSize >= 1 and kernelSize % 2 == 1
    padding = int((kernelSize-1)/2)  #Make sure the output tensors for each channel are also 4x4 by adjusting the padding 
    net = [modelUtils.WSConv2d(inCh=inCh, outCh=outCh, kernelSize=kernelSize, stride=stride, padding=padding, gain=1)]
    return net


class Generator(nn.Module):
    """
    Progressive growth of WGAN generator
    Constructor parameters:
    nOutputChannels (int): the number of color channels in the generated image
    resolution (int): the pixel resolution of the image. The image is assumed to be squared
    fmapBase (int): useful parameter to determine the number of channels in each resolution block (see the getNoChannels function)
    fmapDecay (double): parameter that dictates how fast the number of channels decay while increasing the resolution (see getNoChannels function)
    fmapMax (int): maximum number of channels for any layer
    latentSize (int): size of the latent vector. If None, it is initialized to min(fmapBase,fmapMax)
    """
    def __init__(self, nOutputChannels=3, resolution=256, fmapBase=8192, fmapDecay=1.0, fmapMax=512, latentSize=None):
        super().__init__()
        self.fmapBase = fmapBase
        self.fmapDecay = fmapDecay
        self.fmapMax = fmapMax
        if latentSize == None: latentSize = self.getNoChannels(0)
        nBlocks = int(np.log2(resolution))-1                                #4x4 resolution requires 1 blocks, 8x8 requires 2, and so on
        assert resolution == 2**(nBlocks+1) and resolution >= 4
        
        chain = nn.ModuleList()
        toRGBs = nn.ModuleList()
        net = []

        # First block 4x4
        net += [nn.Linear(latentSize,latentSize*4*4)]                       #Make sure that, no matter the size of the latent vector, it can be shaped into 4x4 tensors
        net += [nn.LeakyReLU(negative_slope=0.2)]                           #Compute activation function
        net += [modelUtils.ReshapeLayer([-1, latentSize, 4, 4])]            #Reshape the output as nBatch x nChannels (= latentSize) x 4 x 4. This counts as the first convolution block
        inCh, outCh = latentSize, self.getNoChannels(1)                     #inCh = min(fmapBase,fmapMax), outCh = min(fmapBase/2,fmapMapx)
        net = genConvBlock(net=net, inCh=inCh, outCh=outCh, kernelSize=3)   #Create the second conv2d block

        toRGB = toRGBBlock(inCh=outCh, outCh=nOutputChannels)               #Take the image back to RGB
        chain.append(nn.Sequential(*net))
        toRGBs.append(nn.Sequential(*toRGB))
        
        # Blocks 8x8 (i = 1), 16x16 (i = 2), and so on
        for i in range(1, nBlocks):
            net = [nn.Upsample(scale_factor=2, mode='bilinear')]                #Upsample using bilinear interpolation
            inCh, outCh = self.getNoChannels(i), self.getNoChannels(i+1)        #Takes care of the channels transitions after upsampling
            net = genConvBlock(net=net, inCh=inCh, outCh=outCh, kernelSize=3)   #Add first 2dConv
            inCh, outCh = self.getNoChannels(i+1), self.getNoChannels(i+1)          #Keep the same number of channels between convolutions
            net = genConvBlock(net=net, inCh=inCh, outCh=outCh, kernelSize=3)   #Add sencond 2dConv
            toRGB = toRGBBlock(inCh=outCh, outCh=nOutputChannels)               #Take the image back to RGB
            chain.append(nn.Sequential(*net))                                    
            toRGBs.append(nn.Sequential(*toRGB))
        
        self.net = modelUtils.ProcessGenLevel(chain=chain, toRGBs=toRGBs)

    def getNoChannels(self, stage):
        """
        Get no. of filters based on below formulae
        stage (int): level of the resolution block. In general, an NxN block corresponds to the stage log(N)-1
        """
        return min(int(self.fmapBase / (2.0 ** (stage * self.fmapDecay))), self.fmapMax)

    def forward(self, x, curResLevel, fadeWt=None):
        """
        Forward the generator through the input x
        x (tensor): latent vector
        fadeWt (double): Weight to regularly fade in higher resolution blocks
        """
        return self.net.forward(x, curResLevel, fadeWt)


##############################################################
# Critic
##############################################################

def criticConvBlock(net, inCh, outCh, kernelSize, stride=1,  negSlope=0.2, padding=None):
    """
    This funtion prepends and returns LIST of conv blocks for critic
    net (list): list to append the current block
    inCh (int): the number of channels in the input 
    outCh (int): the number of channels in the output
    kernelSize (int,list): the dimension(s) of the convolution kernel
    stride (int,list): the steps taken in each dimension
    negSlope (int): the slope for the negative values of the input in the leakyReLU
    padding (int, list): padding on each dimension. If none, padding is calculated to keep the size of the input
    """
    assert kernelSize >= 1 and kernelSize % 2 == 1
    if padding == None: padding = int((kernelSize-1)/2)  #Make sure the output tensors for each channel are also 4x4 by adjusting the padding 
    net += [modelUtils.WSConv2d(inCh=inCh, outCh=outCh, kernelSize=kernelSize, stride=stride, padding=padding)]    
    net += [nn.LeakyReLU(negative_slope=negSlope)]
    return net
    

def fromRGBBlock(inCh, outCh, kernelSize=1, stride=1, negSlope=0.2):
    """
    This creates a list containing a preprocessing block (from RGB to ch)
    inCh (int): the number of color channels in the image
    outCh (int): the number of output channels
    kernelSize (int,list): the dimension(s) of the convolution kernel
    stride (int,list): the steps taken in each dimension
    """
    assert kernelSize >= 1 and kernelSize % 2 == 1
    padding = int((kernelSize-1)/2)  #Make sure the output tensors for each channel are also 4x4 by adjusting the padding 
    net = [modelUtils.WSConv2d(inCh=inCh, outCh=outCh, kernelSize=kernelSize, stride=stride, padding=padding)]
    net += [nn.LeakyReLU(negative_slope=negSlope)]
    return net

class Critic(nn.Module):
    """
    Progressive growth of WGAN critic 
    Constructor parameters:
    nInputChannels (int): the number of color channels in the fed image
    resolution (int): the pixel resolution of the image. The image is assumed to be squared
    fmapBase (int): useful parameter to determine the number of channels in each resolution block (see the getNoChannels function)
    fmapDecay (double): parameter that dictates how fast the number of channels decay while increasing the resolution (see getNoChannels function)
    fmapMax (int): maximum number of channels for any layer
    latentSize (int): size of the latent vector. If None, it is initialized to min(fmapBase,fmapMax)
    batchStdDevGroupSize (int): size of the groups the batch is divided to calculate the statistics in the last part of the critic
    """
    def __init__(self, nInputChannels=3, resolution=256, fmapBase=8192, fmapDecay=1.0, fmapMax=512, batchStdDevGroupSize = 4):
        super().__init__()
        self.fmapBase = fmapBase
        self.fmapDecay = fmapDecay
        self.fmapMax = fmapMax
        nBlocks = int(np.log2(resolution))-1
        assert resolution == 2**(nBlocks+1) and resolution >= 4
        
        chain = nn.ModuleList()
        fromRGBs = nn.ModuleList()
        net = []

        # Last block 4x4
        inCh, outCh = self.getNoChannels(1), self.getNoChannels(0)
        fromRGB = fromRGBBlock(inCh=nInputChannels, outCh=inCh)               #Take the RGB image to the number of channels needed in the net
        
        if batchStdDevGroupSize > 1: 
            net.append(modelUtils.BatchStdConcat(batchStdDevGroupSize))
            inCh = inCh + 1

        net = criticConvBlock(net, inCh=inCh, outCh=outCh, kernelSize=3, padding=1)
        inCh, outCh = self.getNoChannels(0), self.getNoChannels(0)      
        net.append(modelUtils.WSConv2d(inCh=inCh, outCh=outCh, kernelSize=4, stride=1, padding=0))
        net.append(nn.LeakyReLU(negative_slope=0.2)) #Now the output is of size batchSize x outCh x 1 x 1. 
        net.append(modelUtils.ReshapeLayer([-1,outCh])) #Get rid of the trivial dimensions. Otherwise the next line will flat the input into a 1D array of size (batchSize x outCh)
        net.append(nn.Linear(outCh,1))     #Return a critic
                
        chain.append(nn.Sequential(*net))
        fromRGBs.append(nn.Sequential(*fromRGB))
        
        # Higher resolution blocks
        for i in range(1,nBlocks):
            inCh, outCh = self.getNoChannels(i+1), self.getNoChannels(i+1)
            fromRGB = fromRGBBlock(inCh=nInputChannels, outCh=inCh)
            net = []
            net = criticConvBlock(net, inCh=inCh, outCh=outCh, kernelSize=3)   #First convolutional block
            inCh, outCh = self.getNoChannels(i+1), self.getNoChannels(i)     #Double the number of channels for the second convolution 
            net = criticConvBlock(net, inCh=inCh, outCh=outCh, kernelSize=3)   #Second convolutional block
            net.append(nn.AvgPool2d(kernel_size=2,stride=2))                   #Downsample
            
            chain.append(nn.Sequential(*net))
            fromRGBs.append(nn.Sequential(*fromRGB))
        
        self.net = modelUtils.ProcessCriticLevel(fromRGBs=fromRGBs, chain=chain)
    
    def getNoChannels(self, stage):
        """
        Get no. of filters based on below formulae
        """
        return min(int(self.fmapBase / (2.0 ** (stage * self.fmapDecay))), self.fmapMax)
    
    def forward(self, x, curResLevel, fadeWt=None):
        return self.net.forward(x, curResLevel, fadeWt)

    def getOutputGradWrtInputs(self, input, curResLevel, fadeWt=None, device='cpu'):
        """
        Return the unrolled gradient matrix of the critic output wrt the input parameters for 
        each example in the input
        (should have a size batchSize x (imageWidth x imageHeight x imageChannels))
        """
        x = input.detach().requires_grad_().to(device=device)
        out = self.net.forward(x, curResLevel=curResLevel, fadeWt=fadeWt).to(device=device)
        ddx = autograd.grad(outputs=out, inputs=x,
                              grad_outputs = torch.ones(out.size(),device=device),
                              create_graph = True, retain_graph=True, only_inputs=True)[0]
        ddx = ddx.view(ddx.size(0), -1)

        return ddx

