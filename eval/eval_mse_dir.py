import time
import sys
import glob
import os
from options.test_options import TestOptions
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import util
from PIL import Image
from tqdm import tqdm
import numpy as np
from skimage.measure import compare_psnr, compare_ssim

# eval
sum_AtoB = 0.0
sum_BtoA = 0.0
sum_AtoBpsnr = 0.0
sum_BtoApsnr = 0.0
sum_AtoBssim = 0.0
sum_BtoAssim = 0.0
AB_path = sorted(glob.glob(os.path.join(sys.argv[1],'test/*')))
for i, path in enumerate(tqdm(AB_path)):
    AB_ = Image.open(path).convert('RGB').resize((512,256))
    name = path.split('/')[-1].split('.')[0]
    A_ = AB_.crop((0, 0, 256, 256))
    B_ = AB_.crop((256, 0, 512, 256))
    fake_A = np.array(Image.open(os.path.join(sys.argv[2], 'B', name+'.jpg')))
    # fake_B = np.array(Image.open(os.path.join(sys.argv[2], 'A', name+'.jpg')))
    real_A = np.array(A_)
    # real_B = np.array(B_)
    # AtoB = ((fake_B - real_B)**2).sum() / (fake_B.size)
    BtoA = ((fake_A - real_A)**2).sum() / (fake_A.size)
    # AtoBpsnr = compare_psnr(real_B, fake_B)
    BtoApsnr = compare_psnr(real_A, fake_A)
    # AtoBssim = compare_ssim(real_B, fake_B, multichannel=True)
    BtoAssim = compare_ssim(real_A, fake_A, multichannel=True)
    # sum_AtoBpsnr += AtoBpsnr
    sum_BtoApsnr += BtoApsnr
    # sum_AtoBssim += AtoBssim
    sum_BtoAssim += BtoAssim
    # sum_AtoB += AtoB
    sum_BtoA += BtoA
# print 'AtoB:'
# print 'mse: ', sum_AtoB / len(AB_path)
# print 'psnr: ', sum_AtoBpsnr / len(AB_path)
# print 'ssim: ', sum_AtoBssim / len(AB_path)
print 'BtoA: '
print 'mse: ', sum_BtoA / len(AB_path)
print 'psnr: ', sum_BtoApsnr / len(AB_path)
print 'ssim: ', sum_BtoAssim / len(AB_path)

