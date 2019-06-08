import time
import os
from options.test_options import TestOptions
from data.data_loader import create_dataloader
from models.models import create_model
from utils.visualizer import Visualizer
from utils import html
from ipdb import set_trace

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.toy_data = False
opt.eval = False
opt.aux = False
data_loader, dataset, dataset_size = create_dataloader(opt)
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
set_trace()
# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path[0])
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
