import os, PIL.Image as Image, numpy as np, torch, cv2
from torch.utils.data import Dataset, DataLoader
from glob import glob
import math

################################################################################
# Util Functions
################################################################################

def arrayToImage(arr):
    """
    Feed in numpy array in the range of -1 to 1 and return PIL image
    """
    return Image.fromarray(((arr*127)+127).astype('uint8'))


def tensorToImage(tensor):
    """
    Convert a flipped channel tensor to a PIL image
    """
    arr = np.transpose(np.array(tensor.detach()), (1,2,0))
    arr[arr>1]=1; arr[arr<-1]=-1
    arr = ((arr + 1) * 127.5).astype('uint8')
    return Image.fromarray(arr)


def arrayToTensor(array):
    """
    Convert numpy array to tensor after transposing (make the RGB channels be the first index) and float 32 conversion  
    """
    return torch.from_numpy(np.transpose(array.astype('float32'), (2, 0, 1)))

def imagePreprocessing(img,res=None,**kwargs):
    """
    Preprocess an image read with opencsv
    """
    # preprocess 
    smallestDim = min(img.shape[:-1]) #Get the smallest dimension (height or width)
    shape = np.array(img.shape[:-1])
    minIndex = np.floor((shape - smallestDim)/2) #Define the coordinate of the new leftmost, lowest pixel
    maxIndex = shape - np.ceil((shape - smallestDim)/2) #Define the coordinate of the new rightmost, highest pixel
    img = img[int(minIndex[0]):int(maxIndex[0]), 
              int(minIndex[1]):int(maxIndex[1])] #Make the image squared by cropping
    img = img[:, :, ::-1].astype('float32') #The image channels are read in the order BGR, so we invert them to get RGB
    if res != None:
        img = cv2.resize(img, (res, res), cv2.INTER_NEAREST) #Match resolution to the ones needed for the training blocks
    img = img/127.5 - 1 #Center around 0
    return img
        

################################################################################
# Dataloader functions
################################################################################

class imageDataset(Dataset):
    """
    This class represents a dataset of preprocessed images of resolution res x res
    """
    def __init__(self, path, res, preprocess=None):
        self.paths = glob(os.path.join(path, '*.jpg'))
        self.paths += glob(os.path.join(path, '*.png'))
        self.res =  res
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        # Get an image element. It is an array of height x width x channels
        img = cv2.imread(self.paths[idx])
        
        if self.preprocess:
            img = self.preprocess(img,res=self.res)

        # convert to tensor
        img = arrayToTensor(img)
    
        return img

def loadData(path, res, batchSize, numWorkers=4, pinMemory=False, preprocess=imagePreprocessing):
    """
    Function to load and preprocess data from path
    """
    dataset = imageDataset(path, res, preprocess = preprocess)
    dataloader = DataLoader(dataset, batch_size = batchSize, num_workers = numWorkers, shuffle = True, drop_last = True, pin_memory = pinMemory)

    return dataloader

      
