import torch.nn as nn, torch, models, dataUtils, numpy as np, os, pdb, PIL.Image as Image
from torch import FloatTensor as FT
from torch.autograd.variable import Variable
from torch.optim import Adam
from datetime import datetime
import config, os, time


def switchTrainable(nNet, status):
    """
    This is used to switch models parameters to trainable or not
    """
    for p in nNet.parameters(): p.requires_grad = status


def modifyLR(optimizer, lr):
    """
    This function will change LR
    """
    for param in optimizer.param_groups:
        param['lr'] = lr

        
def writeFile(path, content, mode):
    """
    This will write content to a give file
    """
    file = open(path, mode)
    file.write(content); file.write('\n')
    file.close()


class Trainer:
    """
    Trainer class with hyperparams, log, train function etc.
    """
    def __init__(self):
        # Paths        
        self.LOG_DIR = config.LOG_DIR
        self.DATA_PATH = config.DATA_PATH
        self.EXP_DIR = config.EXP_DIR if config.EXP_DIR != None else self.createEXP_DIR
        self.modelFname = config.modelFname
        self.preWtsFile = config.preWtsFile

        #CUDA configuration parameters
        self.use_cuda = config.use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        
        # Hyperparams
        self.cLR=config.cLR; self.gLR=config.gLR
        self.latentSize = config.latentSize
        self.batchSizes =  config.batchSizes
        self.resolutions = config.resolutions
        self.startRes = config.startRes
        self.endRes = config.endRes
        self.samplesWhileStable = config.samplesWhileStable
        self.samplesWhileFade = config.samplesWhileFade
        
        # Hyperparams for loss function of critic
        self.lamb = 10 if config.lamb == None else config.lamb
        self.obj = 1 if config.obj == None else config.obj
        self.epsilon = 0.001 if config.epsilon == None else config.epsilon

        # model 
        self.createModels()
        try:
            self.batchSize = self.batchSizes[self.startRes]
        except:
            "WARNING: There is no batch size defined for the starting resolution. Using a batch size of 10"
            self.batchSize = 10

        self.dataloader = dataUtils.loadData(path=self.DATA_PATH, batchSize=self.batchSize, res=self.startRes, pinMemory = self.use_cuda)
        self.genLoss = []
        self.criticLoss = []; self.criticRealLoss = []; self.criticFakeLoss = []
        self.gradientReal = []; self.gradientFake[]
        self.curResLevel = 0
        
        # Pretrained weigths
        self.usePreWts = config.usePreWts
        if self.usePreWts and self.preWtsFile: self.loadPretrainedWts()

        #Log
        self.logStep = config.logStep
                
    def createModels(self):
        """
        This function will create models and their optimizers
        """
        fmapBase = self.endRes*8 if config.fmapBase == None else config.fmapBase
        fmapMax = self.endRes/2 if config.fmapBase == None else config.fmapMax
        fmapDecay = 1.0 if config.fmapDecay == None else config.fmapDecay
        stdDevGroup = 4 if config.stdDevGroup == None else config.stdDevGroup

        self.gen = models.Generator(resolution = self.endRes, fmapBase = fmapBase, fmapMax = fmapMax, fmapDecay = fmapDecay).to(device=self.device)
        self.crit = models.Critic(resolution = self.endRes, fmapBase = fmapBase, fmapMax = fmapMax, fmapDecay = fmapDecay, batchStdDevGroupSize = stdDevGroup).to(device=self.device)
        self.gOptimizer = Adam(self.gen.parameters(), lr = self.gLR, betas=(0.0, 0.99))
        self.cOptimizer = Adam(self.crit.parameters(), lr = self.cLR, betas=(0.0, 0.99))
        
        print('Models Instantiated. # of trainable parameters Critic: %e; Generator: %e' 
              %(sum([np.prod([*p.size()]) for p in self.crit.parameters()]), 
                sum([np.prod([*p.size()]) for p in self.gen.parameters()])))
        print(f'Model training hyperparameters: fmapBase: {fmapBase}; fmapMax: {fmapMax}; fmapDecay: {fmapDecay}')
        
    def createEXP_DIR(self):
        """
        Create experiment directory
        """
        dir = os.path.join(self.LOG_DIR,'ppwgan-log-' + f'{datetime.now():%d-%m-%HH:%MM}/')
        try: os.makedirs(dir)
        except: print('WARNING: Logging in previously created folder')
        writeFile(os.path.join(self.logDir, 'log.txt'), self.logParameters(), 'w')
        return 'ppwgan-log-' + f'{datetime.now():%d-%m-%HH:%MM}/'
    
    def loadPretrainedWts(self):
        """
        Search for weight file in the experiment directory, and loads it if found
        """
        dir = os.path.join(self.LOG_DIR,self.EXP_DIR,self.preWtsFile)
        if os.path.isfile(dir):
            try:
                wtsDict = torch.load(dir, map_location=lambda storage, loc: storage)
                self.crit.load_state_dict(wtsDict['crit']) 
                self.gen.load_state_dict(wtsDict['gen'])
                self.cOptimizer.load_state_dict(wtsDict['cOptimizer'])
                self.gOptimizer.load_state_dict(wtsDict['gOptimizer'])
            except:
                print('ERROR: The weights in {:s} could not be loaded. Proceding from zero...'.format(dir))
        else:
            print('ERROR: The file {:s} does not exist. Proceding from zero...'.format(dir))

    def logParameters(self):
        """
        This function will return hyperparameters and architecture as string
        """
        hyperParams = f'HYPERPARAMETERS - cLR-{self.cLR}|gLR-{self.gLR}|lambda-{self.lamb}'
                      f'|obj-{self.obj}|epsilon-{self.epsilon}|fadeSteps-{self.samplesWhileFade}|stableSteps-{self.samplesWhileStable}'
        architecture = '\n\n' + str(self.crit) + '\n\n' + str(self.gen) + '\n\n'
        print(hyperParams)    
        return hyperParams + architecture
    
    def logTrainingStats(self):
        """
        Print and write mean losses, save images generated
        """
        # Average all stats and log
        genLoss_ = np.mean(self.genLoss[-self.logStep:])
        critLoss_ = np.mean(self.criticLoss[-self.logStep:])
        critRealLoss_ = np.mean(self.criticRealLoss[-self.logStep:])
        critFakeLoss_ = np.mean(self.criticFakeLoss[-self.logStep:])
        stats = (f'{datetime.now():%HH:%MM}| {self.res}| {self.stage}'
                 f'| {self.currentStep} | {genLoss_:.4f}| {critLoss_:.4f}'
                 f'| {critRealLoss_:.4f}| {critFakeLoss_:.4f}| {self.cLR:.2e}| {self.gLR:.2e}')
        print(stats); 
        f = os.path.join(self.LOG_DIR,self.EXP_DIR,'log.txt')
        writeFile(f, stats, 'a')
        
        # Loop through each image and process
        for _ in range(8):    
            # Fake
            z = self.getNoise(1)
            fake = self.gen(x=z, fadeWt=self.fadeWt)
            f = dataUtils.tensorToImage(fake[0])
            
            # real
            self.callDataIteration()
            r = dataUtils.tensorToImage(self.real[0])
            
            try: img = np.vstack((img, np.hstack((f, r))))
            except: img = np.hstack((f, r))

        # save samples
        fname = str(self.res) + '_' + self.stage + '_' + str(self.currentStep) + '.jpg'
        dir = os.path.join(self.LOG_DIR,self.EXP_DIR,fname)
        Image.fromarray(img).save(dir)
    
    def saveModel(self,status='final'):
        """
        Saves model
        """
        fname = f'finalModel_{self.res}x{self.res}_.pth.tar'
        if status == 'checkpoint':
            fname = 'modelCheckpoint_'+str(self.res)+'_'+self.stage+'_'+str(self.currentStep)+'.pth.tar'

        dir = os.path.join(self.LOG_DIR,self.EXP_DIR,fname)
        f = os.path.join(dir)
        torch.save({'crit':self.crit.state_dict(), 'cOptimizer':self.cOptimizer.state_dict(),
                    'gen':self.gen.state_dict(), 'gOptimizer':self.gOptimizer.state_dict()}, 
                   f)    
    
    def callDataIteration(self):
        """
        This function will call the next value of dataiterator
        """        
        # Next Batch
        try: real = self.dataIterator.next()
        except StopIteration: 
            self.dataIterator = iter(self.dataloader)
            real = self.dataIterator.next()
        
        self.real = real.to(device=self.device)
    
    def getNoise(self, bs=None):
        """
        This function will return noise
        """
        if bs == None : 
            try: bs = self.batchSize
            except: bs = 1
        return FT(bs, self.latentSize).normal_().to(device=self.device)
          
    def trainCritic(self,fadeWt):
        """
        Train the critic for one step
        """
        self.cOptimizer.zero_grad()
        switchTrainable(self.crit, True)
        switchTrainable(self.gen, False)

        # real
        real = self.real
        cRealOut = self.crit(x=real.detach(), fadeWt=fadeWt, curResLevel = self.curResLevel)
        critRealLoss_ = torch.mean(cRealOut)
        
        # fake
        self.z = self.getNoise()
        self.fake = self.gen(x=self.z, fadeWt=fadeWt, curResLevel = self.curResLevel)
        fake = self.fake
        cFakeOut = self.crit(x=fake.detach(), fadeWt=fadeWt, curResLevel = self.curResLevel)
        critFakeLoss_ = torch.mean(cFakeOut)

        #Critic loss
        critLoss_ = critFakeLoss_ - critRealLoss_

        critLoss_.backward()

        # gradient penalty
        gradientReal = autograd.grad(outputs=cRealOut, inputs=real,
                              grad_outputs=torch.ones(cRealOut.size()),
                              create_graph=False, retain_graph=False, only_inputs=True)[0]
        gradientReal = gradientReal.view(gradients.size(0), -1)

        gradientFake = autograd.grad(outputs=cFakeOut, inputs=fake,
                              grad_outputs=torch.ones(cFakeOut.size()),
                              create_graph=False, retain_graph=False, only_inputs=True)[0]
        gradientFake = gradientFake.view(gradients.size(0), -1)
        
        critLoss_ = critLoss_ + self.lamb*((gradientReal.norm(2,dim=1)-self.obj)**2).mean() 
        critLoss_ = critLoss_ + self.lamb*((gradientFake.norm(2,dim=1)-self.obj)**2).mean()

        #Drift loss
        critLoss_ = critLoss_ + self.epsilon*((cRealOut.norm(2))

        ; self.cOptimizer.step()
        return critLoss_.item(), critRealLoss_.item(), critFakeLoss_.item(), gradientReal.item(), gradientFake.item()
    
    def trainGenerator(self,fadeWt):
        """
        Train Generator for 1 step
        """
        self.gOptimizer.zero_grad()
        switchTrainable(self.gen, True)
        switchTrainable(self.crit, False)
        
        self.z = self.getNoise()
        self.fake = self.gen(x=self.z, fadeWt=fadeWt, curResLevel = self.curResLevel
        genCritLoss_ = self.crit(x=self.fake, fadeWt=fadeWt, curResLevel = self.curResLevel
        
        genLoss_ = -genCritLoss_
        genLoss_.backward(); self.gOptimizer.step()
        return genLoss_.item()
    
    def train(self):
        """
        Train function 
        """ 
        samplesPerResolution = self.samplesWhileStable + self.samplesWhileFade #How many training examples are shown for the training of each resolution
        startResLevel = self.resolutions.index(self.startRes) #The index of the starting resolution in the resolutions list
        endResLevel = self.resolutions.index(self.endRes)+1   #The index of the ending resolution in the resolutions list
        totalSteps = self.samplesWhileStable #The first resolution doesn't need the fade steps
        totalSteps = totalSteps + samplesPerResolution*(len(self.resolutions[startResLevel:endResLevel])-1) #For the other resolutions, we have fading and stable steps
        currentStep = config.currentStep if config.currentStep != None else 0 #Initialize the current training step
        
        assert isinstance(currentStep,int), 'ERROR: if different than None, currentStep should be a nonnegative integer' 
        assert currentStep >= 0, 'ERROR: if different than None, currentStep should be a nonnegative integer' 
        assert currentStep < totalSteps, f'ERROR: the current step is larger than the total number of training steps! {currentStep} > {totalSteps}'

        print('Starting training...')        
        print('Time   |res |stage|It        |gLoss  |cLoss |cRLoss |cFLoss |cLR      |gLR      ')
        
        # loop over training steps
        while currentStep < totalSteps:
            
            #Formula that gets the current resolution index
            curResLevel = startResLevel+int((currentStep+self.samplesWhileFade)/samplesPerResolution) 
            
            self.curResLevel = curResLevel
            self.res = self.resolutions[curResLevel]
            self.batchSize = self.batchSizes[self.res]
            
            isFadeStage = (curResLevel > startResLevel) #If we are in the starting resolution, there is not fading
            stepInCurrentRes = ( (currentStep - self.samplesWhileStable) % samplesPerResolution )
            isFadeStage = isFadeStage and ( stepInCurrentRes < self.samplesWhileFade )
            fadeWt = 0

            self.stage = 'stable'

            if isFadeStage:

                self.stage = 'fade'

                #Define the fading weight
                fadeWt = float(stepInCurrentRes+1)/self.samplesWhileFade

                # load new dl if stage is fade or we have loaded data 
                self.dataloader = dataUtils.loadData(path=self.DATA_PATH, batchSize=self.batchSize, res=self.res, pinMemory = self.use_cuda)
                self.dataIterator = iter(self.dataloader)

            #Get batch of training data
            self.callDataIteration()

            # Train Critic
            critLoss_, critRealLoss_, critFakeLoss_, gradientReal_, gradientFake_ = self.trainCritic(fadeWt)
            self.criticLoss.append(critLoss_); self.criticRealLoss.append(critRealLoss_); self.criticFakeLoss.append(critFakeLoss_)
            self.gradientReal.append(gradientReal_); self.gradientFake.append(gradientFake_)

            # Train Gen
            genLoss_ = self.trainGenerator(fadeWt)
            self.genLoss.append(genLoss_)

            # log
            if currentStep % self.logStep == 0 :
                self.currentStep = currentStep
                self.logTrainingStats()
            
            saveModel = False

            if config.saveModel:
                #save model if we finished training the sable stages
                saveModel = ( currentStep == ( (curResLevel+1)*self.samplesWhileStable + curResLevel*self.samplesWhileFade - 1 ) )
                #or if we finished training the fading stages
                saveModel = saveModel | ( currentStep == ( (curResLevel+1)*(self.samplesWhileStable + self.samplesWhileFade) - 1 ) )
            
            if saveModel: 
                self.currentStep = currentStep
                self.saveModel(status='checkpoint')

            currentStep = currentStep + 1

        self.currentStep = currentStep
        self.stage = 'stable'
        self.saveModel(status='final')
            

