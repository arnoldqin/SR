import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import numpy as np
import random

class UnalignedProgDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform_A = get_transform(opt, data='A', resample=Image.NEAREST)
        self.transform_B = get_transform(opt, data='B', resample=Image.NEAREST)

        self.random = not opt.serial_batches
        
        self.train_epoch = opt.train_epoch
        self.trans_epoch = opt.trans_epoch
        self.trans_count = 1
        self.start_res = opt.startSize
        if opt.epoch_count == 0:
            self.current_res = opt.startSize
        else:
            self.current_res = opt.startSize * (2 ** int(opt.epoch_count / (opt.train_epoch + opt.trans_epoch)))
        self.alpha = 0.0
        
        self.trans_status = []
        for i in range(int(np.log(opt.fineSize / opt.startSize)/np.log(2))):
            self.trans_status += [False] * self.train_epoch
            self.trans_status += [True] * self.trans_epoch
        self.trans_status += [False] * ((self.train_epoch+self.trans_epoch+1)*2)

    def tick(self, steps, epoch_size):
        prev_epoch = int((steps-1)/ epoch_size)
        current_epoch = int(steps / epoch_size)
        if self.trans_status[current_epoch]:
            if not self.trans_status[prev_epoch]:
                print "Enter trans status, current trans %d -> %d"%(self.current_res, self.current_res*2)
            self.alpha = float(self.trans_count) / (epoch_size * self.trans_epoch)
            self.trans_count += 1
        else:
            if self.trans_status[prev_epoch]:
                print "End trans status, current trans %d"%(self.current_res*2)
                self.alpha = 0.0
                self.trans_count = 1
                self.current_res *= 2
                
    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        if self.random:
            index_B = random.randint(0, self.B_size - 1)
        else:
            index_B = index % self.B_size
        B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB').resize((self.opt.loadSize ,self.opt.loadSize), Image.BICUBIC)
        B_img = Image.open(B_path).convert('RGB').resize((self.opt.loadSize ,self.opt.loadSize), Image.BICUBIC)
        
        if random.random() > self.alpha:
            A_img = A_img.resize((self.current_res, self.current_res), Image.NEAREST)
            B_img = B_img.resize((self.current_res, self.current_res), Image.NEAREST)
        else:
            A_img = A_img.resize((self.current_res*2, self.current_res*2), Image.NEAREST)
            B_img = B_img.resize((self.current_res*2, self.current_res*2), Image.NEAREST)

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedProgDataset'
