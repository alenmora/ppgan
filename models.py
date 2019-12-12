import modelUtils, torch.nn as nn, numpy as np

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
    if padding == None: padding = int((kernelsize-1)/2)  #Make sure the output tensors for each channel are also 4x4 by adjusting the padding 
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
    padding = int((kernelsize-1)/2)  #Make sure the output tensors for each channel are also 4x4 by adjusting the padding 
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
        nBlocks = int(np.log2(resolution))-1                                #4x4 resolution requires 0 (loop) blocks, 8x8 requires 1, and so on
        assert resolution == 2**(nBlocks+1) and resolution >= 4
        
        chain = nn.ModuleList()
        toRGBs = nn.ModuleList()
        net = []

        # First block 4x4
        net += [nn.Linear(latentSize,self.getNoChannels(0)*4*4)]            #Make sure that, no matter the size of the latent vector, it can be shaped into 4x4 tensors
        net += [nn.LeakyReLU(negative_slope=0.2)]                           #Compute activation function
        net += [modelUtils.ReshapeLayer([4, 4])]                            #Reshape it. This counts as the first convolution block
        inCh, outCh = self.getNoChannels(0), self.getNoChannels(1)          #inCh = min(fmapBase,fmapMax), outCh = min(fmapBase/2,fmapMapx)
        net = genConvBlock(net=net, inCh=outCh, outCh=outCh, kernelSize=3)  #Create the second conv2d block

        toRGB = toRGBBlock(inCh=outCh, outCh=nOutputChannels)               #Take the image back to RGB
        chain.append(nn.Sequental(*net))
        toRGBs.append(nn.Sequental(*toRGB))
        
        # Blocks 8x8 and up
        for i in range(2, nBlocks):
            net = [nn.Upsample(scale_factor=2, mode='bilinear')]                #Upsample using bilinear interpolation
            inCh, outCh = self.getNoChannels(i-1), self.getNoChannels(i)        #Takes care of the channels transitions after upsampling
            net = genConvBlock(net=net, inCh=inCh, outCh=outCh, kernelSize=3)   #Add first 2dConv
            inCh, outCh = self.getNoChannels(i), self.getNoChannels(i)          #Keep the same number of channels between convolutions
            net = genConvBlock(net=net, inCh=inCh, outCh=outCh, kernelSize=3)   #Add sencond 2dConv
            toRGB = toRGBBlock(inCh=outCh, outCh=nOutputChannels)               #Take the image back to RGB
            chain.append(nn.Sequental(*net))                                    
            toRGBs.append(nn.Sequental(*toRGB))
        
        self.net = modelUtils.ProcessGenLevel(chain=chain, toRGBs=toRGBs)

    def getNoChannels(self, stage):
        """
        Get no. of filters based on below formulae
        stage (int): level of the resolution block. In general, an NxN block corresponds to the stage log(N)-1
        """
        return min(int(self.fmapBase / (2.0 ** (stage * self.fmapDecay))), self.fmapMax)

    def forward(self, x, fadeWt=None):
        """
        Forward the generator through the input x
        x (tensor): latent vector
        fadeWt (double): Weight to regularly fade in higher resolution blocks
        """
        return self.net.forward(x, fadeWt)


##############################################################
# Critic
##############################################################

def criticConvBLock(net, inCh, outCh, kernelSize, stride=1,  negSlope=0.2, padding=None):
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
    if padding == None: padding = int((kernelsize-1)/2)  #Make sure the output tensors for each channel are also 4x4 by adjusting the padding 
    net.insert(nn.LeakyReLU(negative_slope=negSlope))
    net.insert(0,modelUtils.WSConv2d(inCh=inCh, outCh=outCh, kernelSize=kernelSize, stride=stride, padding=padding))
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
    padding = int((kernelsize-1)/2)  #Make sure the output tensors for each channel are also 4x4 by adjusting the padding 
    net = [modelUtils.WSConv2d(inCh=inCh, outCh=outCh, kernelSize=kernelSize, stride=stride, padding=padding)]
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
        #Second convolution block (last operation is preprended first)
        inCh, outCh = self.getNoChannels(0), self.getNoChannels(0)      
        net.insert(0,nn.LeakyReLU(negative_slope=0.2))                                                   #Calculate activation function
        net.insert(0,modelUtils.WSConv2d(inCh=inCh, outCh=outCh, kernelSize=4, stride=1, padding=0))     #Unroll to a vector of size min(fmapBase,fmapMax) x 1 x 1

        #The input channels for the first convolution block depend on if we calculate the statistics layer or not
        inCh, outCh = self.getNoChannels(1), self.getNoChannels(0)
        if batchStdDevGroupSize > 1: inCh = inCh+1    
        
        net = criticConvBLock(net, inCh=inCh, outCh=outCh, kernelSize=3, padding=1)   #First convolution block.
        
        #Calculate the statistics and properly define inCh and outCh in case we don't enter the for loop
        inCh, outCh = self.getNoChannels(1), self.getNoChannels(1)                    
        if batchStdDevGroupSize > 1: net.insert(0,modelUtils.BatchStdConcat(batchStdDevGroupSize))

        fromRGB = fromRGBBlock(inCh=nInputChannels, outCh=inCh)               #Take the RGB image to the number of channels needed in the net
        chain.append(nn.Sequential(*net))
        fromRGBs.append(nn.Sequential(*fromRGB))
        
        # Higher resolution blocks
        for i in range(nBlocks, 1, -1):
            net.insert(0,nn.AvgPool2d(kernel_size=2,stride=2))                 #Downsize by averaging
            inCh, outCh = self.getNoChannels(i), self.getNoChannels(i-1)       #Double the number of channels for the second convolution 
            net = criticConvBLock(net, inCh=inCh, outCh=outCh, kernelSize=3)   #Second convolutional block (since criticConvBlock preprends)
            inCh, outCh = self.getNoChannels(i), self.getNoChannels(i)         #Keep the number of channels constant for the first convolution
            net = criticConvBlock(net, inCh=inCh, outCh=outCh, kernelSize=3)   #First convolutional block (since criticConvBlock preprends)
            
            fromRGB = fromRGBBlock(inCh=nInputChannels, outCh=inCh)
            chain.append(nn.Sequential(*net))
            fromRGBs.append(nn.Sequential(*fromRGB))
        
        self.net = modelUtils.ProcessCriticLevel(fromRGBs=fromRGBs, chain=chain)
    
    def getNoChannels(self, stage):
        """
        Get no. of filters based on below formulae
        """
        return min(int(self.fmapBase / (2.0 ** (stage * self.fmapDecay))), self.fmapMax)
    
    def forward(self, x, fadeWt=None):
        return self.net.forward(x, fadeWt)

