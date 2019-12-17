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
        #CUDA configuration parameters
        self.useCuda = config.useCuda and torch.cuda.is_available()
        self.device = torch.device('cuda') if self.useCuda else torch.device('cpu')
        
        # Hyperparams
        self.cLR=config.cLR; self.gLR=config.gLR
        self.latentSize = int(config.latentSize) if config.latentSize != None else int(config.endRes/2)
        self.batchSizes =  config.batchSizes
        self.resolutions = list(config.batchSizes.keys())
        self.startRes = int(config.startRes)
        self.endRes = int(config.endRes)
        self.samplesWhileStable = int(config.samplesWhileStable)
        self.samplesWhileFade = int(config.samplesWhileFade)
        
        # Hyperparams for loss function of critic
        self.lamb = 10 if config.lamb == None else config.lamb
        self.obj = 1 if config.obj == None else config.obj
        self.epsilon = 0.001 if config.epsilon == None else config.epsilon

        # model 
        self.createModels()
        try:
            self.batchSize = int(self.batchSizes[self.startRes])
        except:
            "WARNING: There is no batch size defined for the starting resolution. Using a batch size of 10"
            self.batchSize = 10

        # Paths        
        self.LOG_DIR = config.LOG_DIR
        self.DATA_PATH = config.DATA_PATH
        self.EXP_DIR = self.createEXP_DIR(config.EXP_DIR) if config.EXP_DIR != None else self.createEXP_DIR()
        self.preWtsFile = config.preWtsFile

        #data loading
        self.dataloader = dataUtils.loadData(path=self.DATA_PATH, batchSize=self.batchSize, res=self.startRes, pinMemory = self.useCuda)
        self.dataIterator = iter(self.dataloader)
        
        #monitoring parameters
        self.genLoss = []
        self.criticLoss = []; self.criticRealLoss = []; self.criticFakeLoss = []
        self.gradientLoss = []; self.critGradShape = []; self.driftLoss = []
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
        fmapBase = int(self.endRes*8) if config.fmapBase == None else int(config.fmapBase)
        fmapMax = int(self.endRes/2) if config.fmapBase == None else int(config.fmapMax)
        fmapDecay = 1.0 if config.fmapDecay == None else config.fmapDecay
        stdDevGroup = 4 if config.stdDevGroup == None else config.stdDevGroup

        self.gen = models.Generator(resolution = self.endRes, fmapBase = fmapBase, fmapMax = fmapMax, fmapDecay = fmapDecay, latentSize = self.latentSize).to(device=self.device)
        self.crit = models.Critic(resolution = self.endRes, fmapBase = fmapBase, fmapMax = fmapMax, fmapDecay = fmapDecay, batchStdDevGroupSize = stdDevGroup).to(device=self.device)
        self.gOptimizer = Adam(self.gen.parameters(), lr = self.gLR, betas=(0.0, 0.99))
        self.cOptimizer = Adam(self.crit.parameters(), lr = self.cLR, betas=(0.0, 0.99))
        
        print('Models Instantiated. # of trainable parameters Critic: %e; Generator: %e' 
              %(sum([np.prod([*p.size()]) for p in self.crit.parameters()]), 
                sum([np.prod([*p.size()]) for p in self.gen.parameters()])))
        print(f'Model training hyperparameters: fmapBase: {fmapBase}; fmapMax: {fmapMax}; fmapDecay: {fmapDecay}; stdDevGroup: {stdDevGroup}; latentSize: {self.latentSize}')
        
    def createEXP_DIR(self, fName=None):
        """
        Create experiment directory
        """
        if fName == None: fName = 'ppwgan-log-' + f'{datetime.now():%d/%m-%HH:%MM}/'
        dir = os.path.join(self.LOG_DIR,fName)
        try: 
            os.makedirs(dir)
            print(f'Created new experiment folder at {dir}')
        except FileExistsError: 
            print(f'Logging into previously created folder {fName}')
        writeFile(os.path.join(dir, 'architecture.txt'), self.logParameters(), 'w')
        return fName
    
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
        hyperParams = (f'HYPERPARAMETERS - cLR-{self.cLR}|gLR-{self.gLR}|lambda-{self.lamb}'
                      f'|obj-{self.obj}|epsilon-{self.epsilon}|fadeSteps-{self.samplesWhileFade}|stableSteps-{self.samplesWhileStable}')
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
        gradLoss_ = np.mean(self.gradientLoss[-self.logStep:])
        critGradShape_ = f'({self.critGradShape[-1][0]} x {self.critGradShape[-1][1]})'
        driftLoss_ = np.mean(self.driftLoss[-self.logStep:])
        stats = f' {datetime.now():%H:%M (%d/%m)}'
        stats = stats + "| {:4d}| {:>6s}".format(self.res,self.stage)
        leadingSpaces = 9+len(str(self.logStep))-len(str(int(self.currentStep/self.logStep)))
        stats = stats + "|"+leadingSpaces*" "+str(int(self.currentStep/self.logStep))
        stats = stats + "| {:9.4f}| {:9.4f}".format(genLoss_,critLoss_)
        stats = stats + "| {:9.4f}| {:9.4f}".format(critRealLoss_,critFakeLoss_)
        stats = stats + "| {:9.4f}| {:>10s}| {:9.4f}".format(gradLoss_, critGradShape_, driftLoss_)
        print(stats); 
        f = os.path.join(self.LOG_DIR,self.EXP_DIR,'log.txt')
        writeFile(f, stats, 'a')
        
        # Loop through each image and process
        for _ in range(8):    
            # Fake
            z = self.getNoise(1)
            fake = self.gen(x=z, curResLevel=self.curResLevel, fadeWt=self.fadeWt)
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
        fName = f'finalModel_{self.res}x{self.res}_.pth.tar'
        if status == 'checkpoint':
            fName = 'modelCheckpoint_'+str(self.res)+'_'+self.stage+'_'+str(self.currentStep)+'.pth.tar'

        dir = os.path.join(self.LOG_DIR,self.EXP_DIR,fName)
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
          
    def trainCritic(self):
        """
        Train the critic for one step
        """
        self.cOptimizer.zero_grad()
        switchTrainable(self.crit, True)
        switchTrainable(self.gen, False)

        # real
        real = self.real.requires_grad_(True)
        cRealOut = self.crit(x=real, fadeWt=self.fadeWt, curResLevel = self.curResLevel)
        critRealLoss_ = -1*cRealOut.mean()
        
        # fake
        self.z = self.getNoise()
        self.fake = self.gen(x=self.z, fadeWt=self.fadeWt, curResLevel = self.curResLevel)
        fake = self.fake
        cFakeOut = self.crit(x=fake.detach(), fadeWt=self.fadeWt, curResLevel = self.curResLevel)
        critFakeLoss_ = cFakeOut.mean()

        # gradient penalty
        alpha = torch.rand(self.batchSize, 1, 1, 1).to(device=self.device) 
        interpols = (alpha*real + (1-alpha)*fake).to(device=self.device)
        gradInterpols = self.crit.getOutputGradWrtInputs(interpols, curResLevel = self.curResLevel, fadeWt=self.fadeWt, device=self.device)
        gradLoss_ = self.lamb*((gradInterpols.norm(2,dim=1)-self.obj)**2).mean()

        #Drift loss
        driftLoss_ = self.epsilon*((cRealOut**2).mean()) + self.epsilon*((cFakeOut**2).mean())

        #Final loss
        critLoss_ = critRealLoss_ + critFakeLoss_ + driftLoss_ + gradLoss_ + driftLoss_

        critLoss_.backward(); self.cOptimizer.step()
        return critLoss_.item(), critRealLoss_.item(), critFakeLoss_.item(), gradLoss_.item(), gradInterpols.shape, driftLoss_.item()
    
    def trainGenerator(self):
        """
        Train Generator for 1 step
        """
        self.gOptimizer.zero_grad()
        switchTrainable(self.gen, True)
        switchTrainable(self.crit, False)
        
        self.z = self.getNoise()
        self.fake = self.gen(x=self.z, fadeWt=self.fadeWt, curResLevel = self.curResLevel)
        genCritLoss_ = self.crit(x=self.fake, fadeWt=self.fadeWt, curResLevel = self.curResLevel).mean()
        
        genLoss_ = -1*genCritLoss_
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
        totalSteps = int(totalSteps) #casting for sake of safeness
        currentStep = config.startStep if config.startStep != None else 0 #Initialize the current training step
        
        assert isinstance(currentStep,int), 'ERROR: if different than None, currentStep should be a nonnegative integer' 
        assert currentStep >= 0, 'ERROR: if different than None, currentStep should be a nonnegative integer' 
        assert currentStep < totalSteps, f'ERROR: the current step is larger than the total number of training steps! {currentStep} > {totalSteps}'

        print('Starting training...')        
        print(f'time and date |res  |stage  |iter (x{self.logStep}) |genLoss   |critLoss  |cRealLoss |cFakeLoss |gradLoss  |gradShape  |driftLoss ')
        print('|'.join(['-'*14,'-'*5,'-'*7,'-'*(9+len(str(self.logStep))),'-'*10,'-'*10,'-'*10,'-'*10,'-'*10,'-'*11,'-'*10]))
        
        # loop over training steps
        while currentStep < totalSteps:
            
            #Formula that gets the current resolution index
            curResLevel = startResLevel+int((currentStep+self.samplesWhileFade)/samplesPerResolution) 
            
            self.curResLevel = curResLevel
            self.res = self.resolutions[curResLevel]
            self.batchSize = int(self.batchSizes[self.res])
            
            isFadeStage = (curResLevel > startResLevel) #If we are in the starting resolution, there is not fading
            stepInCurrentRes = int( (currentStep - self.samplesWhileStable) % samplesPerResolution )
            isFadeStage = isFadeStage and ( stepInCurrentRes < self.samplesWhileFade )
            self.fadeWt = 0

            self.stage = 'stable'

            if isFadeStage:

                self.stage = 'fade'
                #Define the fading weight
                self.fadeWt = float(stepInCurrentRes+1)/self.samplesWhileFade

                # load new dl if stage is fade or we have loaded data 
                self.dataloader = dataUtils.loadData(path=self.DATA_PATH, batchSize=self.batchSize, res=self.res, pinMemory = self.useCuda)
                self.dataIterator = iter(self.dataloader)

            #Get batch of training data
            self.callDataIteration()

            # Train Gen (evaluate critic)
            genLoss_ = self.trainGenerator()
            self.genLoss.append(genLoss_)

            # Train Critic
            critLoss_, critRealLoss_, critFakeLoss_, gradientLoss_, critGradShape_, driftLoss_  = self.trainCritic()
            self.criticLoss.append(critLoss_); self.criticRealLoss.append(critRealLoss_); self.criticFakeLoss.append(critFakeLoss_)
            self.gradientLoss.append(gradientLoss_); self.critGradShape.append(critGradShape_); self.driftLoss.append(driftLoss_)

            # log
            if currentStep % self.logStep == 0 :
                self.currentStep = currentStep
                self.logTrainingStats()
            
            if config.saveModelEvery:
                if currentStep > 0 and ((currentStep % config.saveModelEvery) == 0):
                    self.currentStep = currentStep
                    self.saveModel(status='checkpoint')
            
            currentStep = currentStep + 1

        self.currentStep = currentStep
        self.stage = 'stable'
        self.saveModel(status='final')
            

