import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pdb


class PixelNormalization(nn.Module):
    """
    This is the per pixel normalization layer. This will devide each x, y by channel root mean square
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        return x / (torch.mean(x**2, dim=1, keepdim=True) + 1e-8) ** 0.5

    def __repr__(self):
        return self.__class__.__name__ + '(eps = %s)' % (self.eps)


class WSConv2d(nn.Module):
    """
    This is the wt scaling conv layer layer. Initialize with N(0, scale). Then 
    it will multiply the scale for every forward pass
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
        nn.init.normal_(self.conv.weight)
        nn.init.constant_(self.bias, val=0)
        
        self.name = '(inp = %s)' % (self.conv.__class__.__name__ + str(convShape))
        
    def forward(self, x):
        output = self.conv(x) * self.wtScale + self.bias.view(1, self.bias.shape[0], 1, 1)
        return output 

    def __repr__(self):
        return self.__class__.__name__ + self.name

    
class BatchStdConcat(nn.Module):
    """
    Add std to last layer group of disc to improve variance
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
    
    
class ProcessGenLevel(nn.Module):
    """
    Based on the fade wt, this module will use relevant conv layer levels to return generated image
    """
    def __init__(self, chain=None, toRGBs=None):
        super().__init__()
        self.chain = chain
        self.toRGBs = toRGBs

    def forward(self, x, curResLevel, fadeWt):

        for level in range(curResLevel):
                x = self.chain[level](x)

        if (fadeWt == 0):  #If fadeWt is zero we are at a stable stage, so just apply the last resolution layer
            
            x = self.chain[curResLevel](x)
            x = self.toRGBs[curResLevel](x)

            return x
        
        else:   #Otherwise, we are in a fade stage, and we need the output from the previous resolution
        
            prev_x = x #Get the output for the previous resolution
            prev_x = self.toRGBs[curResLevel-1](prev_x) #Transform it to RGB
            prev_x = F.interpolate(prev_x, scale_factor=2, mode='bilinear', align_corners=True) #Upsample it (upsample function is deprecated)

            x = self.chain[curResLevel](x) #Compute the output for the current resolution
            x = self.toRGBs[curResLevel](x) #Transform it to RGB       
        
            return fadeWt*x + (1-fadeWt)*prev_x #Return their interpolation

class ProcessCriticLevel(nn.Module):
    """
    Based on the fade wt, this module will use relevant conv layer levels to return generated image
    """
    def __init__(self, fromRGBs=None, chain=None):
        super().__init__()
        self.fromRGBs = fromRGBs
        self.chain = chain

    def forward(self, x, curResLevel, fadeWt):

        y = self.fromRGBs[curResLevel](x) #Get the input formated from the current level
        for level in range(curResLevel,-1,-1): #Start by applying the highest resolution layer and then go down to the most simple one
            y = self.chain[level](y)
            
            if fadeWt == 0 : return y #If fadeWt is zero we are in a stable stage and don't need anything else

        #Otherwise, we are in a fade stage, and we need the output from the previous resolution
        prev_y = F.avg_pool2d(x, kernel_size=2, stride=2)   #Since the resolution of the input image is twice as the previous one, we downscale
        prev_y = self.fromRGBs[curResLevel-1](prev_y)       #Use the proper RGB to channels filter
        for level in range(curResLevel-1,-1,-1):            #Apply all the filters, from the highest resolution one to the lowest
            prev_y = self.chain[level](prev_y)       
        
        return fadeWt*y + (1-fadeWt)*prev_y #Return their interpolation
    
class ReshapeLayer(nn.Module):
    """
    Reshape latent vector layer
    """
    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
