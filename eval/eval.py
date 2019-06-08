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
#opt.dataroot = './datasets/cityscapes/'
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

#opt.name = 'final_city_skip-featonly_block-6-gp'
old_opt.eval = False
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

if old_opt.alpha_gate == 'simp':
        model.netG_A.gate.mask = 1
        model.netG_B.gate.mask = 1

for i, data in enumerate(tqdm(dataset)):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    fake_B = Image.fromarray(util.tensor2im(model.fake_B.data.cpu()))
    fake_A = Image.fromarray(util.tensor2im(model.fake_A.data.cpu()))
    name_A = data['A_paths'][0].split('/')[-1]
    name_B = data['B_paths'][0].split('/')[-1]
    fake_B.save(os.path.join(save_dir_A, name_A))
    fake_A.save(os.path.join(save_dir_B, name_B))
