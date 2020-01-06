import torch as torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image

class dataLoader:
    def __init__(self, config):
        self.data_path = config.data_path
        self.batch_table = {4:32, 8:32, 16:32, 32:24, 64:16, 128:16, 256:16, 512:8, 1024:8} 
        self.imsize = int(config.startRes)
        self.batchSize = int(self.batch_table[config.startRes])
        self.num_workers = 0

    def renew(self, resl):
        print(f'[*] Renew dataloader configuration, load data from {self.data_path} with resolution {resl}x{resl}')
        
        self.batchSize = int(self.batch_table[resl])
        self.imsize = int(resl)
        self.dataset = ImageFolder(
                    root=self.data_path,
                    transform=transforms.Compose(   [
                                                    transforms.Resize(size=(self.imsize,self.imsize), interpolation=Image.NEAREST),
                                                    transforms.ToTensor(),
                                                    ]))

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batchSize,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory = torch.cuda.is_available()
        )

    def __iter__(self):
        return iter(self.dataloader)
    
    def __next__(self):
        return next(self.dataloader)

    def __len__(self):
        return len(self.dataloader.dataset)
   
    def get_batch(self):
        dataIter = iter(self.dataloader)
        return next(dataIter)[0].mul(2).add(-1)         # pixel range [-1, 1]

    def get(self, n = None):
        if n == None: n = self.batchSize

        x = self.get_batch()
        for i in range(n // self.batchSize):
            torch.nn.cat([x, self.get_batch()], 0)
        
        return x[:n]
