import numpy as np
import os
import ntpath
import time
from . import util
from . import html
from pdb import set_trace as st
from tensorboardX import SummaryWriter
import torch
import torchvision.utils as vutils
from datetime import datetime

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.use_html = opt.isTrain and not opt.no_html
        self.name = opt.name
        self.tensorboard = SummaryWriter(log_dir=os.path.join(opt.checkpoints_dir, 'logdir', self.name+'_'+datetime.now().strftime('%m-%d_%H-%M-%S')))

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, iters):
        img_list = [torch.Tensor(image_numpy.transpose([2,0,1]).astype('f')) for label, image_numpy in visuals.items()]
        self.tensorboard.add_image(self.name+'/eval_img', vutils.make_grid(img_list, nrow=len(img_list)//2, normalize=True, scale_each=True), iters)

        if self.use_html: # save images to a html file
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=256)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, iters, errors, title):
        for k in errors.keys():
            self.tensorboard.add_scalar('%s/%s'%(title,k), errors[k], iters)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t, name=''):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(name, message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_params_hist(self, net_params, iters):
        for net_name, params in net_params:
            for name, param in params:
                self.tensorboard.add_histogram('%s/%s'%(net_name,name), param, iters)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0][0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=256)
