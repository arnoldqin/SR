import os.path
import torchvision.transforms as transforms
from data.unaligned_dataset import UnalignedDataset
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import numpy as np

class UnalignedAttrDataset(UnalignedDataset):
    def initialize(self, opt):
        UnalignedDataset.initialize(self, opt)

    def __getitem__(self, index):
        sample = UnalignedDataset.__getitem__(self, index)
        A_path = sample['A_paths']
        A_attr = int(A_path.split('/')[-1].split('_')[0])
        assert(A_attr == 1 or A_attr == 0)
        B_path = sample['B_paths']
        B_attr = int(B_path.split('/')[-1].split('_')[0])
        assert(B_attr == 1 or B_attr == 0)
        return {'A': sample['A'], 'B': sample['B'],
                'A_attr': np.array(A_attr), 'B_attr': np.array(B_attr),
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedAttrDataset'
