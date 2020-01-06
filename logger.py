import torch
import numpy as np
import utils
import os
from datetime import datetime

class logger:
    """
    Logger class to output net status, images and network snapshots
    """
    def __init__(self, trainer, config):
        self.log_path = utils.createDir(config.log_path)
        self.logStep = int(config.logStep)
        self.saveImageEvery = int(config.saveImageEvery)
        self.saveModelEvery = int(config.saveModelEvery)
        self.trainer = trainer
        self.device = trainer.device
        self.latentSize = int(config.latentSize)
        self.resumeTraining = (config.resumeTraining != None)
        
        self.dontLog = config.deactivateLog
        
        self.samplesWhileStable = int(config.samplesWhileStable)
        self.samplesWhileFade = int(config.samplesWhileFade)

        self.nCritPerGen = int(config.nCritPerGen)

        self.cLR = config.cLR
        self.gLR = config.gLR

        self.lamb = config.lamb                     #lambda 
        self.obj = config.obj                       #objective value (1-GP)
        self.epsilon = config.epsilon               #epsilon (drift loss)
        self.delta = config.delta                   #delta (TV regularization)
        self.lambk = config.lambk                   #Learning rate for kt (BEGAN)
        self.gamma = config.gamma                   #ratio between expected values for the real and fake losses in the critic (BEGAN)
        self.criticLossFunc = config.criticLossFunc
        self.extraLossTerm = config.extraLossTerm
        
        self.netStatusHeaderShown = False
        self.arch_file = 'architecture.txt'
        self.log_file = 'netStatus.txt'
        
        #monitoring parameters
        self.genLoss = 0
        self.criticLoss = 0; self.criticRealLoss = 0; self.criticFakeLoss = 0
        self.extraLoss = 0; self.driftLoss = 0

        self.logCounter = -1
        self.ncAppended = 0
        self.ngAppended = 0

        self.snapCounter = 0
        self.imgCounter = 0

    def logArchitecture(self):
        """
        This function will print hyperparameters and architecture and save the in the log directory under the architecture.txt file
        """
        if self.dontLog:
            return

        cstFcn = f'Cost function model: {self.criticLossFunc}\n'
        if self.criticLossFunc == 'BEGAN': cstFcn + f'** lambda_k = {self.lambk}| gamma = {self.gamma}\n'
        
        hyperParams = (f'HYPERPARAMETERS - cLR = {self.cLR}|gLR = {self.gLR}|Using uncentered GP with: lambda = {self.lamb}'
                      f' and obj = {self.obj}|epsilon = {self.epsilon}|fadeSteps = {self.samplesWhileFade}|stableSteps = {self.samplesWhileStable}\n')
        if self.extraLossTerm == '0GP':
            hyperParams = (f'HYPERPARAMETERS - cLR = {self.cLR}|gLR = {self.gLR}|Using 0-centered GP with: lambda = {self.lamb}'
                          f'|epsilon = {self.epsilon}|fadeSteps = {self.samplesWhileFade}|stableSteps = {self.samplesWhileStable}\n')
        elif self.extraLossTerm == 'TVP':
            hyperParams = (f'HYPERPARAMETERS - cLR = {self.cLR}|gLR = {self.gLR}|Using TV regularization with: lambda = {self.lamb}'
                          f'|delta = {self.delta}|fadeSteps = {self.samplesWhileFade}|stableSteps = {self.samplesWhileStable}\n')


        architecture = '\n' + str(self.trainer.crit) + '\n\n' + str(self.trainer.gen) + '\n\n'
        
        print(cstFcn+hyperParams)

        f = os.path.join(self.log_path, self.arch_file)

        utils.writeFile(f, cstFcn+hyperParams+architecture, 'w')

    def appendGLoss(self, gloss):
        """
        This function will append the generator loss to the genLoss list
        """
        self.startLogging() #Log according to size of appendGLoss, so call the function when appending
        if self.dontLog:
            return
        self.genLoss = (self.genLoss + gloss).detach().requires_grad_(False)
        self.ngAppended =+ 1

    def appendCLoss(self, closs, crloss, cfloss, cextraloss, cdriftloss):
        """
        This function will append the critic training output to the critic lists
        """
        if self.dontLog:
            return
        self.criticLoss = (self.criticLoss + closs).detach().requires_grad_(False)
        self.criticRealLoss = (self.criticRealLoss + crloss).detach().requires_grad_(False)
        self.criticFakeLoss = (self.criticFakeLoss + cfloss).detach().requires_grad_(False)
        self.extraLoss = (self.extraLoss + cextraloss).detach().requires_grad_(False)
        self.driftLoss = (self.driftLoss + cdriftloss).detach().requires_grad_(False)
        self.ncAppended =+ 1

    def startLogging(self):
        
        snapCounter = int(self.trainer.imShown) // self.saveModelEvery
        imgCounter = int(self.trainer.imShown) // self.saveImageEvery

        if snapCounter > self.snapCounter:
            self.saveSnapshot()
            self.snapCounter = snapCounter
        
        if imgCounter > self.imgCounter:
            self.outputPictures()
            self.imgCounter = imgCounter

        if self.dontLog:
            return

        logCounter = int(self.trainer.imShown) // self.logStep
        
        if logCounter > self.logCounter:
            self.logNetStatus()

            #Release memory
            self.genLoss = 0
            self.criticLoss = 0
            self.criticRealLoss = 0
            self.criticFakeLoss = 0
            self.extraLoss = 0
            self.driftLoss = 0
            self.ncAppended = 0
            self.ngAppended = 0

            torch.cuda.empty_cache()

            self.logCounter = logCounter
        
    def logNetStatus(self):
        """
        Print and write mean losses and current status of net (resolution, stage, images shown)
        """
        if self.netStatusHeaderShown == False:
            colNames = f'time and date |res  |bs   |stage  |fadeWt |iter           |genLoss   |critLoss  |cRealLoss |cFakeLoss |extraLoss |driftLoss '
            sep = '|'.join(['-'*14,'-'*5,'-'*5,'-'*7,'-'*7,'-'*15,'-'*10,'-'*10,'-'*10,'-'*10,'-'*10,'-'*10])
            print(colNames)
            print(sep)

            f = os.path.join(self.log_path,self.log_file)  #Create a new log file
            
            if not self.resumeTraining:
                utils.writeFile(f, colNames, 'w')

            utils.writeFile(f, sep, 'a')

            self.netStatusHeaderShown = True       
      
        res = int(self.trainer.res)
        stage = self.trainer.stage
        fadeWt = self.trainer.fadeWt
        imShown = int(self.trainer.imShown)
        batchSize =self.trainer.batchSize

        # Average all stats and log
        gl = self.genLoss.item()/self.ngAppended if self.ngAppended != 0 else 0.
        cl = self.criticLoss.item()/self.ncAppended if self.ncAppended != 0 else 0.
        crl = self.criticRealLoss.item()/self.ncAppended if self.ncAppended != 0 else 0.
        cfl = self.criticFakeLoss.item()/self.ncAppended if self.ncAppended != 0 else 0.
        el = self.extraLoss.item()/self.ncAppended if self.ncAppended != 0 else 0.
        dl = self.driftLoss.item()/self.ncAppended if self.ncAppended != 0 else 0.
        
        stats = f' {datetime.now():%H:%M (%d/%m)}'
        stats = stats + "| {:4d}| {:4d}| {:>6s}| {:6.4f}".format(res,batchSize,stage,fadeWt)
        leadingSpaces = 15-len(str(imShown))
        stats = stats + "|"+leadingSpaces*" "+str(imShown)
        stats = stats + "| {:9.4f}| {:9.4f}".format(gl,cl)
        stats = stats + "| {:9.4f}| {:9.4f}".format(crl,cfl)
        stats = stats + "| {:9.4f}| {:9.5f}".format(el, dl)
    
        print(stats); 
        f = os.path.join(self.log_path,self.log_file)
        utils.writeFile(f, stats, 'a')

    def saveSnapshot(self, title=None):
        """
        Saves model snapshot
        """
        if title == None:
            if self.trainer.stage == 'stable':
                title = f'modelCheckpoint_{self.trainer.res}x{self.trainer.res}_{self.trainer.imShownInRes}_{self.trainer.latentSize}.pth.tar'

                path = os.path.join(self.log_path,title)
                torch.save({'crit':self.trainer.crit.state_dict(), 'cOptimizer':self.trainer.cOptimizer.state_dict(),
                    'gen':self.trainer.gen.state_dict(), 'gOptimizer':self.trainer.gOptimizer.state_dict()}, 
                   path)    

        else:
            title = title+'.pth.tar'
            path = os.path.join(self.log_path,title)
            torch.save({'crit':self.trainer.crit.state_dict(), 'cOptimizer':self.trainer.cOptimizer.state_dict(),
                    'gen':self.trainer.gen.state_dict(), 'gOptimizer':self.trainer.gOptimizer.state_dict()}, 
                   path) 

    def outputPictures(self, size=8):
        """
        outputs real and fake picture samples
        """
        real = self.trainer.getReals(size)
        fake, _ = self.trainer.getFakes(size)
        stacked = torch.cat([real, fake], dim = 0).cpu()
        fName = '_'.join([str(self.trainer.res),str(self.trainer.stage),str(self.trainer.imShownInRes)+'.jpg'])
        path = os.path.join(self.log_path,fName)
        utils.saveImage(stacked, path, nrow = real.size(0))