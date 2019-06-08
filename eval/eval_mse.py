import time
import os
from options.test_options import TestOptions
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import util
from PIL import Image
from tqdm import tqdm
from skimage.measure import compare_psnr, compare_ssim

opt = TestOptions().parse(stop_write=True)
opt_def = TrainOptions().parse(stop_write=True)
save_path = os.path.join('./checkpoints', opt.name)
old_opt = util.load_opt(save_path, opt_def)
old_opt.continue_train = True
old_opt.test = True
opt.fineSize = old_opt.fineSize
opt.loadSize = old_opt.loadSize
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.dataset_mode = 'aligned'
#opt.dataroot = './datasets/cityscapes/'
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

#opt.name = 'final_city_skip-featonly_block-6-gp'
model = create_model(old_opt)

save_dir_A = os.path.join(opt.results_dir, opt.name, 'A')
try:
    os.makedirs(save_dir_A)
except OSError:
    pass
save_dir_B = os.path.join(opt.results_dir, opt.name, 'B')
try:
    os.makedirs(save_dir_B)
except OSError:
    pass
# eval
sum_AtoB = 0.0
sum_BtoA = 0.0
sum_AtoBpsnr = 0.0
sum_BtoApsnr = 0.0
sum_AtoBssim = 0.0
sum_BtoAssim = 0.0
for i, data in enumerate(tqdm(dataset)):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    fake_B = util.tensor2im(model.fake_B.data.cpu())
    fake_A = util.tensor2im(model.fake_A.data.cpu())
    real_A = util.tensor2im(model.real_A.data.cpu())
    real_B = util.tensor2im(model.real_B.data.cpu())
    name_A = data['A_paths'][0].split('/')[-1]
    name_B = data['B_paths'][0].split('/')[-1]
    assert(name_A == name_B)
    AtoB = ((fake_B - real_B)**2).sum() / (fake_B.size)
    BtoA = ((fake_A - real_A)**2).sum() / (fake_A.size)
    AtoBpsnr = compare_psnr(real_B, fake_B)
    BtoApsnr = compare_psnr(real_A, fake_A)
    AtoBssim = compare_ssim(real_B, fake_B, multichannel=True)
    BtoAssim = compare_ssim(real_A, fake_A, multichannel=True)
    sum_AtoBpsnr += AtoBpsnr
    sum_BtoApsnr += BtoApsnr
    sum_AtoBssim += AtoBssim
    sum_BtoAssim += BtoAssim
    sum_AtoB += AtoB
    sum_BtoA += BtoA
    # fake_B.save(os.path.join(save_dir_A, name_A))
    # fake_A.save(os.path.join(save_dir_B, name_B))
print 'AtoB:'
print 'mse: ', sum_AtoB / len(dataset)
print 'psnr: ', sum_AtoBpsnr / len(dataset)
print 'ssim: ', sum_AtoBssim / len(dataset)
print 'BtoA: '
print 'mse: ', sum_BtoA / len(dataset)
print 'psnr: ', sum_BtoApsnr / len(dataset)
print 'ssim: ', sum_BtoAssim / len(dataset)

