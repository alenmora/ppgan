import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import utils
import pdb

class ReshapeLayer(nn.Module):
    """
    Reshape latent vector layer
    """
    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class PixelNormalization(nn.Module):
    """
    This is the per pixel normalization layer. This will divide each x, y by channel root mean square
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        return x / (torch.mean(x**2, dim=1, keepdim=True) + self.eps) ** 0.5

    def __repr__(self):
        return self.__class__.__name__ + '(eps = %s)' % (self.eps)


class WSConv2d(nn.Module):
    """
    This is the wt scaling conv layer layer. Initialize with N(0, 1). Then 
    it will multiply the conv output by gain/kernelSize for every forward pass
    """
    def __init__(self, inCh, outCh, kernelSize, stride, padding, gain=np.sqrt(2)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=inCh, out_channels=outCh, kernel_size=kernelSize, stride=stride, padding=padding)
        
        # new bias to use after wscale
        self.bias = self.conv.bias
        self.conv.bias = None
        
        # calc wt scale
        convShape = list(self.conv.weight.shape)
        fanIn = np.prod(convShape[1:]) # Leave out # of op filters
        self.wtScale = gain/np.sqrt(fanIn)
        
        # init
        nn.init.normal_(self.conv.weight)*self.wtScale
        nn.init.constant_(self.bias, val=0)
        
        self.name = '(inp = %s)' % (self.conv.__class__.__name__ + str(convShape))
        
    def forward(self, x):
        output = self.conv(x)*self.wtScale + self.bias.view(1, self.bias.shape[0], 1, 1)
        return output 

    def __repr__(self):
        return self.__class__.__name__ + self.name

    
class BatchStdConcat(nn.Module):
    """
    Add std to last layer group of critic to improve variance
    """
    def __init__(self, groupSize = 4):
        super().__init__()
        self.groupSize = groupSize

    def forward(self, x):
        shape = list(x.size())                                              # NCHW - Initial size
        xStd = x.view(self.groupSize, -1, shape[1], shape[2], shape[3])     # GMCHW - split minbatch into M groups of size G (= groupSize)
        xStd -= torch.mean(xStd, dim=0, keepdim=True)                       # GMCHW - Subract mean over groups
        xStd = torch.mean(xStd ** 2, dim=0, keepdim=False)                  # MCHW - Calculate variance over groups
        xStd = (xStd + 1e-08) ** 0.5                                        # MCHW - Calculate std dev over groups
        xStd = torch.mean(xStd.view(xStd.shape[0], -1), dim=1, keepdim=True).view(-1, 1, 1, 1)
                                                                            # M111 - Take mean over CHW
        xStd = xStd.repeat(self.groupSize, 1, shape[2], shape[3])           # N1HW - Expand to same shape as x with one channel 
        output = torch.cat([x, xStd], 1)
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + '(Group Size = %s)' % (self.groupSize)

class equalizedLinear(nn.Module):
    """
    Equalizes the learning rate for the weights by scaling them using the normalization constant
    from He's initializer
    """
    def __init__(self, inCh, outCh, gain=np.sqrt(2)):
        super().__init__()
        self.linear = nn.Linear(inCh, outCh)
        
        # new bias to use after wscale
        self.bias = self.linear.bias
        self.linear.bias = None
        
        # calc wt scale
        self.wtScale = gain/np.sqrt(inCh+outCh)
        
        # init
        nn.init.normal_(self.linear.weight)*self.wtScale
        nn.init.constant_(self.bias, val=0)
        
        self.name = '(inp = %s)' % (self.linear.__class__.__name__ + str(self.linear.weight.shape))
        
    def forward(self, x):
        output = self.linear(x)*self.wtScale + self.bias.view(1, self.bias.shape[0])
        return output 

    def __repr__(self):
        return self.__class__.__name__ + self.name   
    
class ProcessGenLevel(nn.Module):
    """
    Based on the fade wt, this module will use relevant conv layer levels to return generated image
    """
    def __init__(self, chain=None, toRGBs=None):
        super().__init__()
        self.chain = chain
        self.toRGBs = toRGBs

    def forward(self, x, curResLevel, fadeWt=0):

        for level in range(curResLevel):
                x = self.chain[level](x)
        
        if fadeWt < 1:   #We are in a fade stage
            prev_x = x #Get the output for the previous resolution
            prev_x = self.toRGBs[curResLevel-1](prev_x) #Transform it to RGB
            prev_x = F.interpolate(prev_x, scale_factor=2, mode='nearest') #Upsample it (upsample function is deprecated)

            x = self.chain[curResLevel](x) #Compute the output for the current resolution
            x = self.toRGBs[curResLevel](x) #Transform it to RGB       
        
            x = fadeWt*x + (1-fadeWt)*prev_x
        
        else:
            x = self.chain[curResLevel](x) #Compute the output for the current resolution
            x = self.toRGBs[curResLevel](x) #Transform it to RGB       
        
        return x 

class ProcessCriticLevel(nn.Module):
    """
    Based on the fade wt, this module will use relevant conv layer levels to return generated image
    """
    def __init__(self, fromRGBs=None, chain=None):
        super().__init__()
        self.fromRGBs = fromRGBs
        self.chain = chain

    def forward(self, x, curResLevel, fadeWt=0):

        if fadeWt < 1:
            prev_x = F.avg_pool2d(x, kernel_size=2, stride=2)   #Since the resolution of the input image is twice as the previous one, we downscale
            prev_x = self.fromRGBs[curResLevel-1](prev_x)       #Use the proper RGB to channels filter
            x = self.fromRGBs[curResLevel](x) #Get the input formated from the current level
            x = self.chain[curResLevel](x)    #Process the top level   

            for level in range(curResLevel-1,-1,-1): #Apply the rest of the levels, from top to bottom
                x = self.chain[level](x)
                prev_x = self.chain[level](prev_x)
        
                x = fadeWt*x + (1-fadeWt)*prev_x

        else:
            x = self.fromRGBs[curResLevel](x) #Get the input formated from the current level

            for level in range(curResLevel,-1,-1): #Apply the rest of the levels, from top to bottom
                x = self.chain[level](x)
        
        return x
