# PGGAN configuration options

import argparse, time
parser = argparse.ArgumentParser('PGGAN')

############################
#  Paths
############################

parser.add_argument('--log_path', type=str, default='./pggan/log/')       # Folder were the training output logs are stored
parser.add_argument('--data_path', type=str, default='./data/')           # Folder were the training data is stored
parser.add_argument('--preWtsFile', type=str, default=None)               # File to get the pretrained weights from

############################
#  Logging
############################

parser.add_argument('--deactivateLog', action='store_true')         #If passed, there will be no logging, except for the final weights
parser.add_argument('--saveModelEvery', type=int, default=100000)    #(Approx) Number of images shown before saving a checkpoint of the model
parser.add_argument('--saveImageEvery', type=int, default=50000)     #(Approx) Number of images shown before generating a set of images and saving them in the log directory
parser.add_argument('--logStep', type=int, default=2500)            #(Approx) Number of images shown before writing a log in the log directory
parser.add_argument('--nChannels', type=int, default=3)             #Number of channels in the training and in the generated images

############################
#  Training parameters
############################

parser.add_argument('--gLR', type=float, default=0.001)                    #Generator learning rate
parser.add_argument('--gLR_decay', type=float, default=0.5)                #Generator learning rate decay constant (applied every resolution change)
parser.add_argument('--gLR_wdecay', type=float, default=0.)                #Generator weight decay constant
parser.add_argument('--cLR', type=float, default=0.001)                    #Critic learning rate
parser.add_argument('--cLR_decay', type=float, default=0.5)                 #Critic learning rate decay constant (applied every resolution change)
parser.add_argument('--cLR_wdecay', type=float, default=0.)                #Critic weight decay constant
parser.add_argument('--nCritPerGen', type=int, default=1)                  #Number of critic training loops per generator training loop
parser.add_argument('--stdDevGroup', type=int, default=8)                  #Size of the groups to calculate the std dev in the last block of the critic
parser.add_argument('--latentSize', type=int, default=128)                 #Size of the latent vector
parser.add_argument('--smoothing', type=float, default=0.997)              #Smoothing factor for the generator
parser.add_argument('--samplesWhileStable', type=int, default=750000)      #Number of images shown while in the stable stage of training
parser.add_argument('--samplesWhileFade', type=int, default=750000)        #Number of images shown while in the fading stage of training
parser.add_argument('--startRes', type=int, default=4)                     #Starting image resolution
parser.add_argument('--endRes', type=int, default=64)                      #Final image resolution
parser.add_argument('--gOptimizer_betas', type=str, default='0.0 0.99')    #Generator adam optimizer beta parameters
parser.add_argument('--cOptimizer_betas', type=str, default='0.0 0.99')    #Critic adam optimizer beta parameters
parser.add_argument('--lamb', type=float, default=10)                      #Weight of the extra loss term in the critic loss function
parser.add_argument('--obj', type=float, default=450)                      #Objective value for the gradient norm in GP regularization (arXiv:1704.00028v3)
parser.add_argument('--epsilon', type=float, default=1e-3)                 #Weight of the loss term related to the magnitud of the loss function for the critic
parser.add_argument('--delta', type=float, default=5)                      #Delta term for TV penalization (arXiv:1812.00810v1)
parser.add_argument('--lambk', type=float, default=1e-3)                   #Learning rate for the parameter kt (BEGAN loss function)
parser.add_argument('--gamma', type=float, default=0.8)                    #ratio between expected values for the real and fake losses in the critic (BEGAN)
parser.add_argument('--paterm', nargs='?')                                 #Include a repelling regularizer term in the generator (arXiv:1609.03126v4). The user should specify if the term is as described in the original paper (by passing False to the flag), or centered around the distance of the inputs (by passing True)
parser.add_argument('--lambg', type=float, default=1)                      #Weiht of the pulling-away term in the generator
parser.add_argument('--unrollCritic',nargs='?')                            #For an integer value n greater than 1, it unrolls the discriminator n steps (arXiv:1611.02163v4)

############################
#  Network parameters
############################

# from the equation nchannels = min(fmapBase/(2**(nBlock*fmapDecay)), fmapMax)
parser.add_argument('--fmapBase', type=int, default=2048)                                #Parameter to calculate the number of channels in each block
parser.add_argument('--fmapMax', type=int, default=256)                                  #Parameter to calculate the number of channels in each block
parser.add_argument('--fmapDecay', type=float, default=1.)                               #Parameter to calculate the number of channels in each block
parser.add_argument('--criticLossFunc', choices=['WGAN','BEGAN'], default='WGAN')        #Which loss function use for the critic. Options are 'WGAN' (arXiv:1701.07875) or 'BEGAN' (arXiv:1703.10717v4)
parser.add_argument('--extraLossTerm', choices=['PGGAN','0GP','TVP'], default='PGGAN')   #Which extra loss to add to the loss function. The default is a combination of GP and drifting penalization, as used in the original paper. For 0GP, see arXiv:1902.03984v1

############################
#  Filter for resizing input images
############################

parser.add_argument('--inputFilter', type=str, default='NEAREST')       #Filter to use in the dataLoader when resizing the input images

############################
#  Resume training
############################
parser.add_argument('--resumeTraining', nargs='*')  #Resumes a previous training. The user must specify the current resolution and the number of images already shown in said resolution

##Parse and save configuration
config, _ = parser.parse_known_args()
