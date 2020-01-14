import torch.nn as nn 
import torch 
import numpy as np
from torch.optim import Adam, lr_scheduler
from datetime import datetime
from config import config 
import utils
import models 
from dataLoader import dataLoader
from logger import logger
import os
import math
import copy
  
class Trainer:
    """
    Trainer class with hyperparams, log, train function etc.
    """
    def __init__(self, config):

        #CUDA configuration parameters
        self.useCuda = torch.cuda.is_available()
        if torch.cuda.is_available():
            self.useCuda = True
            self.device = torch.device('cuda')
            print('Using CUDA...')
        else:
            self.useCuda = False
            self.device = torch.device('cpu')
        
        self.config = config
        
        # Hyperparams
        self.nCritPerGen = int(config.nCritPerGen)
        assert self.nCritPerGen > 0, 'ERROR: The number of critic training loops per generator loop should be an integer >= 1'
        self.cLR=config.cLR; self.gLR=config.gLR
        self.latentSize = int(config.latentSize)
        self.endRes = int(config.endRes)
        self.startRes = int(config.startRes)
        startResLevel, endResLevel = int(np.log2(self.startRes)), int(np.log2(self.endRes))
        assert self.startRes == 2**(startResLevel), 'The initial resolution must be a power of 2, between 4 and 1024'
        assert self.endRes == 2**(endResLevel) and self.endRes <= 1024, 'The final resolution must be a power of 2, between 4 and 1024'
        assert self.startRes < self.endRes, 'The initial resolution must be smaller than the final resolution'
        powRange = np.linspace(startResLevel, endResLevel, (endResLevel-startResLevel + 1), endpoint=True)
        self.resolutions = 2**powRange
        self.samplesWhileStable = int(config.samplesWhileStable)
        self.samplesWhileFade = int(config.samplesWhileFade)
                
        #Training parameters
        self.lazyReg = config.lazyRegularization if config.lazyRegularization else 1
        self.imShown = 0
        self.imShownInRes = 0
        self.curResLevel = 0
        self.res = self.startRes
        self.stage = 'stable'
        self.fadeWt = 1.
        self.kUnroll = 0
        if config.unrollCritic:
            self.kUnroll = int(config.unrollCritic)
        
        assert self.kUnroll >= 0, f'The unroll parameter is less than zero ({self.kUnroll})'

        # Loss function of critic
        self.lamb = config.lamb              #lambda 
        self.obj = config.obj                #objective value (1-GP)
        self.epsilon = config.epsilon        #epsilon (drift loss)
        self.delta = config.delta            #delta (TV regularization)
        self.kt = 0
        self.lambk = config.lambk            #Learning rate for kt (BEGAN)
        self.gamma = config.gamma            #ratio between expected values for the real and fake losses in the critic (BEGAN)
        self.criticLossFunc = config.criticLossFunc
        self.extraLossTerm = config.extraLossTerm

        #Loss function of generator
        self.paterm = int(config.paterm)
        self.lambg = config.lambg

        # models
        self.createModels()

        # Optimizers
        assert config.gLR_decay > 0 and config.gLR_decay <= 1, 'The decay constant for the learning rate of the generator must be a constant between [0, 1]'
        self.gLR_decay =config.gLR_decay

        betas = config.gOptimizer_betas.split(' ')
        beta1, beta2 = float(betas[0]), float(betas[1])
        assert config.gLR_wdecay >= 0 and config.gLR_wdecay <= 1, 'The weight decay constant for the generator must be a constant between [0, 1]'
        self.gOptimizer = Adam(filter(lambda p: p.requires_grad, self.gen.parameters()), lr = self.gLR, betas=(beta1, beta2), weight_decay=config.gLR_wdecay)

        self.glr_scheduler = lr_scheduler.LambdaLR(self.gOptimizer,lambda epoch: self.gLR_decay)

        assert config.cLR_decay > 0 and config.cLR_decay <= 1, 'The decay constant for the learning rate of the critic must be a constant between [0, 1]'
        self.cLR_decay =config.cLR_decay

        assert config.cLR_wdecay >= 0 and config.cLR_wdecay <= 1, 'The weight decay constant for the critic must be a constant between [0, 1]'
        betas = config.cOptimizer_betas.split(' ')
        beta1, beta2 = float(betas[0]), float(betas[1])
        self.cOptimizer = Adam(filter(lambda p: p.requires_grad, self.crit.parameters()), lr = self.cLR, betas=(beta1, beta2), weight_decay=config.cLR_wdecay)

        self.clr_scheduler = lr_scheduler.LambdaLR(self.cOptimizer,lambda epoch: self.cLR_decay)
                
        # Paths        
        self.preWtsFile = config.preWtsFile
        
        if config.resumeTraining:
            self.resumeTraining(config)
        
        elif self.preWtsFile: self.loadPretrainedWts()
        

        #data loading
        self.dataLoader = dataLoader(config)
        self.dataLoader.renew(self.res)
        self.batchSize = self.dataLoader.batchSize
        
        #Log
        self.logger = logger(self, config)
        self.logger.logArchitecture()

        print(f'The trainer has been instantiated.... Starting step: {self.imShown}. Start resolution: {self.res}. Final resolution: {self.endRes}')
        
    def resumeTraining(self, config):
        """
        Resumes the model training, if a valid pretrained weights file is given, and the 
        starting resolution and number of images shown for the current resolution are correctly
        specified
        """
        if not self.loadPretrainedWts():
            print('Could not load weights for the pretrained model. Starting from zero...')
            return

        res, imShownInRes = config.resumeTraining
        res, imShownInRes = int(res), int(imShownInRes)
        curResLevel = 0
            
        curResLevel = np.squeeze(np.argwhere(self.resolutions == res))
        
        if curResLevel.size == 0:
            print('The current resolution is not a power of 2 between 4 and 1024. Proceeding from zero...')
            return
 
        if curResLevel > 0 and imShownInRes < self.samplesWhileFade:
            print(f'The training resuming can only proceed from stable phases. However, the number of images already shown ({imShownInRes}) '
                  f'is less than the number of images shown in the fading stage ({self.samplesWhileFade}). Proceeding from zero...')

        else:
            if curResLevel > 0:
                if imShownInRes > self.samplesWhileStable+self.samplesWhileFade:
                    print(f'The number of images already shown ({imShownInRes}) is higher than the number of images shown for one resolution '
                          f'training ({self.samplesWhileStable+self.samplesWhileFade})! Proceeding from zero...')
                else:
                    self.imShown = curResLevel*(self.samplesWhileFade + self.samplesWhileStable) + imShownInRes
                    self.imShownInRes = imShownInRes
                    self.res = res
                    self.curResLevel = int(curResLevel)
                    for i in range(curResLevel):
                        self.clr_scheduler.step()
                        self.glr_scheduler.step()
            else:
                if imShownInRes > self.samplesWhileStable:
                    print(f'The number of images already shown ({imShownInRes}) is higher than the number of images shown for the first ' 
                          f'resolution ({self.samplesWhileStable})! Proceeding from zero...')
                else:
                    self.imShown = imShownInRes
                    self.imShownInRes = imShownInRes
                    self.res = res
                    self.curResLevel = int(curResLevel)
                        
    def createModels(self):
        """
        This function will create models and their optimizers
        """
        self.gen = models.Generator(self.config).to(self.device)
        self.crit = models.Critic(self.config).to(self.device)
        
        print('Models Instantiated. # of trainable parameters Critic: %e; Generator: %e' 
              %(sum([np.prod([*p.size()]) for p in self.crit.parameters()]), 
                sum([np.prod([*p.size()]) for p in self.gen.parameters()])))
        
    def loadPretrainedWts(self):
        """
        Search for weight file in the experiment directory, and loads it if found
        """
        dir = self.preWtsFile
        if os.path.isfile(dir):
            try:
                wtsDict = torch.load(dir, map_location=lambda storage, loc: storage)
                self.crit.load_state_dict(wtsDict['crit']) 
                self.gen.load_state_dict(wtsDict['gen'])
                self.cOptimizer.load_state_dict(wtsDict['cOptimizer'])
                self.gOptimizer.load_state_dict(wtsDict['gOptimizer'])
                print(f'Loaded pre-trained weights from {dir}')
                return True
            except:
                print(f'ERROR: The weights in {dir} could not be loaded. Proceding from zero...')
                return False
        else:
            print(f'ERROR: The file {dir} does not exist. Proceding from zero...')    
        
        return False

    def getReals(self, n = None):
        """
        Returns n real images
        """ 
        if n == None: n = self.batchSize
        return self.dataLoader.get(n).to(device = self.device)

    def getFakes(self, n = None):
        """
        Returns n fake images and their latent vectors
        """ 
        if n == None: n = self.batchSize
        z = utils.getNoise(bs = n, latentSize = self.latentSize, device = self.device)
        return self.gen(z, curResLevel = self.curResLevel, fadeWt=self.fadeWt), z

    def getBatchReals(self):
        """
        Returns a batch of real images
        """ 
        return self.dataLoader.get_batch().to(device = self.device)

    def getBatchFakes(self):
        """
        Returns a batch of fake images and the latent vector which generated it
        """
        return self.getFakes()
    
    def trainCritic(self):
        """
        Train the critic for one step and store outputs in logger
        """
        self.cOptimizer.zero_grad()
        utils.switchTrainable(self.crit, True)
        utils.switchTrainable(self.gen, False)

        # real
        real = self.getBatchReals()
        cRealOut = self.crit(x=real, curResLevel = self.curResLevel, fadeWt=self.fadeWt)
        critRealLoss_ = cRealOut.mean()
        
        # fake
        fake, _ = self.getBatchFakes()
        cFakeOut = self.crit(x=fake.detach(), curResLevel = self.curResLevel, fadeWt=self.fadeWt)
        critFakeLoss_ = cFakeOut.mean()

        multFactor = 1.0
        if self.criticLossFunc == 'BEGAN': #arXiv:1703.10717v4
            multFactor = self.kt
            self.kt = (self.kt + self.lambk*(self.gamma*critRealLoss_-critFakeLoss_)).detach()
            critFakeLoss_ = -critFakeLoss_
            critRealLoss_ = -critRealLoss_
        
        # original loss
        driftLoss_, extraLoss_ = torch.tensor(0.).to(self.device), torch.tensor(0.).to(self.device)

        # GP and regularizators
        if self.extraLossTerm == 'PGGAN' and self.imShown % self.lazyReg == 0: #Use gradient penalty and drift loss
            # drift loss
            driftLoss_ = self.epsilon*(cRealOut**2).mean()

            #gradient penalty
            alpha = torch.rand(self.batchSize, 1, 1, 1, device=self.device)
            interpols = (alpha*real + (1-alpha)*fake).detach().requires_grad_(True)
            gradInterpols = self.crit.getOutputGradWrtInputs(interpols, curResLevel = self.curResLevel, fadeWt=self.fadeWt, device=self.device)
            shape_ = gradInterpols.shape
            extraLoss_ = self.lamb*((gradInterpols.norm(dim=1)-self.obj)**2).mean()/(self.obj+1e-8)**2

        elif self.extraLossTerm == '0GP' and self.imShown % self.lazyReg == 0: #Use zero-centered gradient penalty arXiv:1902.03984v1
            alpha = torch.rand(self.batchSize, 1, 1, 1, device=self.device)
            interpols = (alpha*real + (1-alpha)*fake).detach().requires_grad_(True)
            gradInterpols = self.crit.getOutputGradWrtInputs(interpols, curResLevel = self.curResLevel, fadeWt=self.fadeWt, device=self.device)
            shape_ = gradInterpols.shape
            extraLoss_ = self.lamb*(gradInterpols.norm(2,dim=1).mean())
        
        elif self.extraLossTerm == 'TVP' and self.imShown % self.lazyReg == 0: #USe TV penalty arXiv:1812.00810v1
            extraLoss_ = self.lamb*(cRealOut-cFakeOut-self.delta).norm(1)

        critLoss_ = multFactor*critFakeLoss_ - critRealLoss_ + driftLoss_ + extraLoss_
        
        critLoss_.backward(); self.cOptimizer.step()

        self.logger.appendCLoss(critLoss_, critRealLoss_, critFakeLoss_, extraLoss_, driftLoss_)
        
    def trainGenerator(self):
        """
        Train Generator for 1 step and store outputs in logger
        """
        self.gOptimizer.zero_grad()
        utils.switchTrainable(self.gen, True)
        utils.switchTrainable(self.crit, False)
        
        fake, latent = self.getBatchFakes()
        genCritLoss_ = self.crit(x=fake, fadeWt=self.fadeWt, curResLevel = self.curResLevel).mean()
        genLoss_ = -genCritLoss_

        if self.paterm != None and self.imShown % self.lazyReg == 0:
            genLoss_ = genLoss_ + self.lambg*self.gen.paTerm(latent, self.curResLevel, self.fadeWt, againstInput = self.paterm)
        
        genLoss_.backward(); self.gOptimizer.step()
        
        self.logger.appendGLoss(genLoss_)

        return fake.size(0)

    def train(self):
        """
        Main train loop
        """ 
        samplesPerResolution = self.samplesWhileStable + self.samplesWhileFade #How many training examples are shown for the training of each resolution
        endResLevel = len(self.resolutions)-1                                  #The index of the ending resolution in the resolutions list
        
        print('Starting training...')   
        self.logger.startLogging() #Start the logging

        # Loop over the first resolution, which only has stable stage. Since each batch shows batchSize images, 
        # we need only samplesWhileStable/batchSize loops to show the required number of images
        if self.curResLevel == 0:
            while self.imShownInRes < self.samplesWhileStable:
                self.doOneTrainingStep() #Increases self.imShownInRes and self.imShown
                
            self.imShownInRes = 0 #Reset the number of images shown at the end (to allow for training resuming)

        # loop over resolutions 8 x 8 and higher
        while self.curResLevel < endResLevel:
            
            #loop over the number of images we need to show for one complete resolution
            while self.imShownInRes < samplesPerResolution:

                self.fadeWt = min(float(self.imShownInRes)/self.samplesWhileFade, 1)
                self.stage = 'fade' if self.fadeWt < 1 else 'stable'

                if self.stage == 'fade' and self.fadeWt == 0: #We just began to fade. So we need to increase the resolutions
                    self.clr_scheduler.step() #Reduce learning rate
                    self.glr_scheduler.step() #Reduce learning rate

                    self.curResLevel = self.curResLevel+1
                    self.res = int(self.resolutions[self.curResLevel])
                    self.dataLoader.renew(self.res)
                    self.batchSize = self.dataLoader.batchSize
                
                self.doOneTrainingStep() #Increases self.imShownInRes and self.imShown
            
            self.imShownInRes = 0 #Reset the number of images shown at the end (to allow for training resuming)

        self.logger.saveSnapshot(f'{self.res}x{self.res}_final_{self.latentSize}')
            
    def doOneTrainingStep(self):
        """
        Performs one train step for the generator, and nCritPerGen steps for the critic
        """ 
        if self.kUnroll:
            for i in range(self.nCritPerGen):
                self.trainCritic()
                if i == 0:
                    self.cBackup = copy.deepcopy(self.crit)
        else:                
            for i in range(self.nCritPerGen):
                self.trainCritic()
        
        shown = self.trainGenerator() #Use the generator training batches to count for the images shown, not the critic
        
        if self.kUnroll:
            self.crit.load(self.cBackup)

        self.imShown = self.imShown + int(shown) 
        self.imShownInRes = self.imShownInRes + int(shown)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True           # boost speed.
    Trainer = Trainer(config)
    Trainer.train()