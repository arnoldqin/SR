import argparse
import os, shutil, glob
from utils import util
import torch
from datetime import datetime

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('-f', type=str, help='dummy arg, for ipython notebook')
        self.parser.add_argument('--debug', action='store_true', help='debug flag')
        self.parser.add_argument('--tanh_out', action='store_true', help='debug flag')
        self.parser.add_argument('--stack_keep_nc', action='store_true', help='debug flag')
        self.parser.add_argument('--stack_imgin', action='store_true', help='debug flag')
        self.parser.add_argument('--stack_simp_conv', action='store_true', help='debug flag')
        self.parser.add_argument('--use_dense', action='store_true', help='debug flag')
        self.parser.add_argument('--downsample_7', action='store_true', help='debug flag')
        self.parser.add_argument('--dataroot', default='./datasets/danbooru_celeba', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--sub_out', type=int, default=3, help='input batch size')
        self.parser.add_argument('--simple_conv', type=int, default=0, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=128, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=128, help='then crop to this size')
        self.parser.add_argument('--CloadSize', type=int, default=256, help='scale images to this size of C')
        self.parser.add_argument('--CfineSize', type=int, default=256, help='then crop to this size of C')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--out_num', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--z_dim', type=int, default=128, help='# of output image channels')
        self.parser.add_argument('--stage', type=int, default=2, help='stage')
        self.parser.add_argument('--not_caffe', action='store_true', help='double gan skip')
        self.parser.add_argument('--half_add', action='store_true', help='double gan skip')
        self.parser.add_argument('--skip', action='store_true', help='double gan skip')
        self.parser.add_argument('--idt', action='store_true', help='double gan skip')
        self.parser.add_argument('--out7', action='store_true', help='double gan skip')
        self.parser.add_argument('--test', action='store_true', help='double gan skip')
        self.parser.add_argument('--skip_feat', action='store_true', help='double gan skip')
        self.parser.add_argument('--feat_only', action='store_true', help='double gan skip')
        self.parser.add_argument('--img_only', action='store_true', help='double gan skip')
        self.parser.add_argument('--img_2ngf', action='store_true', help='double gan skip')
        self.parser.add_argument('--alpha_gate', type=str, default='', help='double gan skip')
        self.parser.add_argument('--not_mono_gate', action='store_true', help='double gan skip')
        self.parser.add_argument('--simple_block', action='store_true', help='double gan skip')
        self.parser.add_argument('--load_pre', action='store_true', help='load premodel')
        self.parser.add_argument('--pre_path', default='' , help='path to premodel')
        self.parser.add_argument('--tune_pre', action='store_true', help='finetune premodel')
        self.parser.add_argument('--lr_pre', type=float, default=0.00002, help='pre lr')
        self.parser.add_argument('--up_pre', action='store_true', help='use upsample pre as input')
        self.parser.add_argument('--keep_pre', action='store_true', help='use upsample pre as input')
        self.parser.add_argument('--dense', action='store_true', help='use denseblock in G')
        self.parser.add_argument('--unet', action='store_true', help='use denseblock in G')
        self.parser.add_argument('--single_D', action='store_true', help='use denseblock in G')
        self.parser.add_argument('--pix2pix_D', action='store_true', help='use denseblock in G')
        self.parser.add_argument('--use_noise', action='store_true', help='test')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--kw', type=int, default=3, help='# of discrim kernal size, for lagecy model')
        self.parser.add_argument('--upsample', default='resize', help='upsampling method, (conv|subpix)')
        self.parser.add_argument('--equalize_input', default='', help='')
        self.parser.add_argument('--use_lrelu', action='store_true', help='finetune premodel')
        self.parser.add_argument('--shift_alpha', action='store_true', help='finetune premodel')
        self.parser.add_argument('--n_downsample', type=int, default=2, help='# of downsamples block')
        self.parser.add_argument('--n_upsample', type=int, default=2, help='# of upsamples block')
        self.parser.add_argument('--max_ngf', type=int, default=256, help='max ngf')
        self.parser.add_argument('--shallow_resblock', action='store_true', help='use a fancy skip in resblock')
        self.parser.add_argument('--n_resblocks', type=int, default=6, help='# of reblocks in resnet G')
        self.parser.add_argument('--n_resblocks_next', type=int, default=2, help='# of reblocks in resnet G')
        self.parser.add_argument('--legacy_D', action='store_true', help='use D that has no downsample block')
        self.parser.add_argument('--which_model_netD', type=str, default='n_layers', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
        self.parser.add_argument('--which_model_netG_pre', type=str, default='resnet_9blocks', help='selects model to use for netG')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--n_layers_D_pre', type=int, default=3, help='only used when stack cycle')
        self.parser.add_argument('--one_out', action='store_true', help='D output size')
        self.parser.add_argument('--feat_len_D', type=int, default=256, help='use when D is one size out, the last fc input size')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        self.parser.add_argument('--model', type=str, default='cycle_gan',
                                 help='chooses which model to use. cycle_gan, pix2pix, test')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')

        self.initialized = True

    def parse(self, stop_write=False):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        # set threads
        torch.set_num_threads(self.opt.nThreads)

        # continue train
        if  self.isTrain and self.opt.continue_train:
            print('Continue train')
            save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            old_opt = util.load_opt(save_path, self.opt)
            old_opt.continue_train = True
            old_opt.name = self.opt.name
            old_opt.epoch_count = self.opt.epoch_count
            self.opt = old_opt

        # load submodel options
        if self.opt.load_pre or self.opt.pre_path != '':
            self.opt.pre = util.load_opt(self.opt.pre_path, self.opt, {'loadSize': self.opt.fineSize / 2, 'fineSize': self.opt.fineSize / 2})

        args = vars(self.opt)

        # print options
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        if not self.isTrain or self.opt.continue_train or stop_write:
            return self.opt

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        self.opt.workdir = expr_dir
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        # save current code to checkpoint dir
        ignore_list = ['etc', 'pkg', 'ref', 'tmp', 'datasets', 'checkpoints']
        if os.path.isdir(os.path.join(self.opt.workdir, 'src')):
            shutil.rmtree(os.path.join(self.opt.workdir, 'src'))
        os.makedirs(os.path.join(self.opt.workdir, 'src'))
        for file in glob.glob('./*'):
            if file.split('/')[-1] not in ignore_list:
                if os.path.isdir(file):
                    shutil.copytree(file, os.path.join(self.opt.workdir, 'src', file.split('/')[-1]))
                else:
                    shutil.copy2(file, os.path.join(self.opt.workdir, 'src'))

        return self.opt
