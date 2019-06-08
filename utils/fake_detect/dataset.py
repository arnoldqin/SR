import os, sys, json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
import copy
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import warnings
warnings.filterwarnings("ignore")

def load_data_dir(dir, label):
    l = glob.glob(dir)
    ret = [(label, i) for i in l]
    return ret

class Dataset(Dataset):
     def __init__(self, root_dir, data_type = 'train', toy_data = False, transform=transforms.ToTensor(), split_task='pass'):
        """
        Args:
            root_dir (string): Dataset dir
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_samples = []
        if data_type == 'train':
            self.data_samples += load_data_dir(os.path.join(root_dir, 'good/*'), 1.0)
            self.data_samples += load_data_dir(os.path.join(root_dir, 'bad/*'), 0.0)
            random.shuffle(self.data_samples)
        elif data_type == 'val':
            self.data_samples += load_data_dir(os.path.join(root_dir, 'good_val/*'), 1.0)
            self.data_samples += load_data_dir(os.path.join(root_dir, 'bad_val/*'), 0.0)
            random.shuffle(self.data_samples)
        else:
            pass

        if toy_data:
            self.data_samples = self.data_samples[:1000]
        print 'Loaded data %s with split %s, total %d samples'%(data_type, split_task, len(self.data_samples))
        self.data_type = data_type
        self.root_dir = root_dir
        self.transform = transform
        self.split_task = split_task

     def __len__(self):
        return len(self.data_samples)

     def __getitem__(self, idx):
        image = Image.open(self.data_samples[idx][-1]).convert('RGB')

        if 'train' in self.data_type or 'val' in self.data_type:
            label = float(self.data_samples[idx][0])
        else:
            label = 0

        if self.transform:
            image = self.transform(image)

        sample = (image, label)

        return sample

