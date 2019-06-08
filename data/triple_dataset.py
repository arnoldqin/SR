import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageOps
import PIL
import random
import tqdm

## TODO support multi-batch training

class TripleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        print('Use dataset from %s'%self.root)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')
  
        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
	self.C_paths = make_dataset(self.dir_C)
        if opt.toy_data:
            self.A_paths = self.A_paths[:100]
            self.B_paths = self.B_paths[:100]
	    self.C_paths = self.C_paths[:100]
	
        self.A_paths = sorted(self.A_paths)
        # self.A_imgs = []
        # for i in tqdm.tqdm(self.A_paths):
                # self.A_imgs.append(Image.open(i).convert('RGB').resize((opt.loadSize, opt.loadSize), Image.BICUBIC))
        self.B_paths = sorted(self.B_paths)
        # self.B_imgs = []
        # for i in tqdm.tqdm(self.B_paths):
                # self.B_imgs.append(Image.open(i).convert('RGB').resize((opt.loadSize, opt.loadSize), Image.BICUBIC))
        self.C_paths = sorted(self.C_paths)
	# self.C_imgs = []
	# for i in tqdm.tqdm(self.C_paths):
		# self.C_imgs.append(Image.open(i).convert('RGB').resize((opt.loadSize, opt.loadSize), Image.BICUBIC))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
	self.C_size = len(self.C_paths)
	
        self.transform_A = get_transform(opt, data='A')
        self.transform_B = get_transform(opt, data='B')
	self.transform_C = get_transform(opt, data='C')
        self.random = not opt.serial_batches

    def __getitem__(self, index):
        B_path = self.B_paths[index % self.B_size]
        index_B = index % self.B_size
        if self.random:
            index_A = random.randint(0, self.A_size - 1)
	    index_C = random.randint(0, self.C_size - 1)		
        else:
            index_A = index % self.A_size
	    index_C = index % self.C_size
        A_path = self.A_paths[index_A]
	C_path = self.C_paths[index_C]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        C_img = Image.open(C_path).convert('RGB')

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
	B_hd = self.transform_C(B_img)
	C = self.transform_C(C_img)
        C_sr = self.transform_B(C_img)
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
            tmp = C[0, ...] * 0.299 + C[1, ...] * 0.587 + C[2, ...] * 0.114
            C = tmp.unsqueeze(0)

        if self.opt.aux:
            A_aux_path = A_path.replace(self.opt.phase+'A', 'auxA')
            B_aux_path = B_path.replace(self.opt.phase+'B', 'auxB')
	    C_aux_path = C_path.replace(self.opt.phase+'C', 'auxC')
            A_aux_img = Image.open(A_aux_path).convert('RGB')
            B_aux_img = Image.open(B_aux_path).convert('RGB')
            C_aux_img = Image.open(C_aux_path).convert('RGB')
            A_aux = self.transform_A(A_img)
            B_aux = self.transform_B(B_img)
            C_aux = self.transform_C(C_img)
            return {'A': A, 'B': B, 'A_aux': A_aux, 'B_aux': B_aux, 'C_aux': C_aux,
                    'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path}
        else:
            return {'A': A, 'B': B, 'C': C, 'C_sr': C_sr, 'B_hd':B_hd,
                    'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path}

    def __len__(self):
        return max(self.A_size, self.B_size, self.C_size)

    def name(self):
        return 'TripleDataset'
