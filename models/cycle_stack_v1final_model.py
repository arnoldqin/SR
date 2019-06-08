import numpy as np
import torch
import os
import random
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import utils.util as util
from utils.util import load_network_with_path
from utils.image_pool import ImagePool
from .base_model import BaseModel
from .cycle_gan_model import CycleGANModel
from . import networks
import sys
from collections import namedtuple
import torchvision.models.vgg as vgg
import torch.utils.model_zoo as model_zoo
import tqdm
from scipy.misc import imresize

class CycleStackv1FinalModel(CycleGANModel):
    def name(self):
        return 'CycleStackv1FinalModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        #self.down_2 = torch.nn.AvgPool2d(2**int((np.log(opt.fineSize) - np.log(opt.pre.fineSize))/np.log(2)))
        #self.up_2 = torch.nn.Upsample(scale_factor=2**int((np.log(opt.fineSize) - np.log(opt.pre.fineSize))/np.log(2)))
        if not opt.idt:
                self.down_2 = torch.nn.AvgPool2d(2)
                self.up_2 = torch.nn.Upsample(scale_factor=2)
        else:
                self.down_2 = torch.nn.AvgPool2d(1)
                self.up_2 = torch.nn.AvgPool2d(1)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, n_upsampling=opt.n_upsample, n_downsampling=opt.n_downsample, side='A', opt=opt)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, n_upsampling=opt.n_upsample, n_downsampling=opt.n_downsample, side='B', opt=opt)
        
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt=opt)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt=opt)
            if 'stack' in opt.which_model_netD and opt.load_pre:
                netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            'n_layers',
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt=opt)
                netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            'n_layers',
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt=opt)
                pre_opt = opt
                for i in range(opt.num_D - 2, -1, -1):
                    util.load_network_with_path(netD_A, 'D_A', opt.pre_path)
                    util.load_network_with_path(netD_B, 'D_B', opt.pre_path)
                    exec('self.netD_A.layer%d = netD_A'%i)
                    exec('self.netD_B.layer%d = netD_B'%i)
                    pre_opt = opt.pre

        print('---------- Networks initialized -------------') 
        networks.print_network(self.netG_A, opt, input_shape=(opt.input_nc, opt.fineSize, opt.fineSize))
        if self.isTrain:
            networks.print_network(self.netD_A, opt, input_shape=(3,opt.fineSize,opt.fineSize))
        print('-----------------------------------------------')

        if not self.isTrain or opt.continue_train:
            print('Continue from ', opt.which_epoch)
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain and not opt.test:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            
            # print('Load VGG 16')
            # vgg_model = vgg.vgg16(pretrained=True)
            # if torch.cuda.is_available():
                # vgg_model.cuda()
            # self.loss_network = networks.VGGLossNetwork(vgg_model)
            # self.loss_network.eval()
            # del vgg_model
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            # if self.opt.cyc_perc:
                # self.criterionCycle = networks.PerceptualLoss(self.loss_network, tensor=self.Tensor)
            # else:
            self.criterionCycle = networks.RECLoss()
            # self.criterionPerc = networks.PerceptualLoss(self.loss_network, tensor=self.Tensor)
            # self.criterionColor = networks.ColorLoss()
            # initialize optimizers
            if opt.tune_pre:
                self.optimizer_G_A = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_G_B = torch.optim.Adam(self.netG_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.alpha_gate != '':
                self.optimizer_G_A = torch.optim.Adam(itertools.chain(
                                             self.netG_A.model_in.parameters()
                                            ,self.netG_A.model_mid.parameters()
                                            ,self.netG_A.model_out.parameters()
                                            ,self.netG_A.gate.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_G_B = torch.optim.Adam(itertools.chain(
                                             self.netG_B.model_in.parameters()
                                            ,self.netG_B.model_mid.parameters()
                                            ,self.netG_B.model_out.parameters()
                                            ,self.netG_B.gate.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G_A = torch.optim.Adam(itertools.chain(
                                             self.netG_A.model_in.parameters()
                                            ,self.netG_A.model_mid.parameters()
                                            ,self.netG_A.model_out.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_G_B = torch.optim.Adam(itertools.chain(
                                             self.netG_B.model_in.parameters()
                                            ,self.netG_B.model_mid.parameters()
                                            ,self.netG_B.model_out.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.d_lr2:
                self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=(opt.lr/2.0), betas=(opt.beta1, 0.999))
                self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=(opt.lr/2.0), betas=(opt.beta1, 0.999))
            else:
                self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G_A)
            self.optimizers.append(self.optimizer_G_B)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

    def tick(self, total_iter, dataset_size):
        if self.opt.alpha_gate == 'simp':
            end_iter = dataset_size * self.opt.alpha_epoch
            if total_iter < end_iter:
                self.netG_A.gate.mask = total_iter / (end_iter)
                self.netG_B.gate.mask = total_iter / (end_iter)
            elif total_iter >= end_iter:
                self.netG_A.gate.mask = 1.0
                self.netG_B.gate.mask = 1.0
                
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.get_fake(self.netG_A, self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.get_fake(self.netG_B, self.real_B)

    def get_fake(self, net, t):
        return net.forward(t, multi_out=True)

    def get_gan_loss(self, pred_fake, l, lambda_adv):
        loss = self.criterionGAN(pred_fake, l, lambda_adv)
        if isinstance(loss, (tuple, list)):
            loss_list = loss[-1]
            loss = loss[0]
        else:
            loss_list = [loss.data[0]]
        return loss, loss_list

    def get_rec_loss(self, rec, real, lambda_rec):
        loss = self.criterionCycle(rec, real, lambda_rec)
        if isinstance(loss, (tuple, list)):
            loss_list = loss[-1]
            loss = loss[0]
        else:
            loss_list = [loss.data[0]]
        return loss, loss_list

   
    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_rec = self.opt.lambda_rec
        lambda_adv = self.opt.lambda_adv
        
        self.optimizer_G_A.zero_grad()
        self.optimizer_G_B.zero_grad()
        # GAN loss
        self.fake_B = self.get_fake(self.netG_A, self.real_A)
        pred_fake = self.netD_A.forward(self.fake_B)
        self.loss_G_A, self.loss_G_A_list = self.get_gan_loss(pred_fake, True, lambda_adv)
        self.fake_A = self.get_fake(self.netG_B, self.real_B)
        pred_fake = self.netD_B.forward(self.fake_A)
        self.loss_G_B, self.loss_G_B_list = self.get_gan_loss(pred_fake, True, lambda_adv)

        # cycle loss
        self.rec_A = self.get_fake(self.netG_B, self.fake_B)
        self.loss_cycle_A, self.loss_cycle_A_list = self.get_rec_loss(self.rec_A, self.real_A, lambda_rec)
        self.rec_B = self.get_fake(self.netG_A, self.fake_A)
        self.loss_cycle_B, self.loss_cycle_B_list = self.get_rec_loss(self.rec_B, self.real_B, lambda_rec)

        # get mask data
        self.post_A = self.netG_A.out
        self.post_B = self.netG_B.out
        if self.opt.alpha_gate != '':
            self.mask_A = self.netG_A.gate.mask
            self.mask_B = self.netG_B.gate.mask

        # aux loss
        self.loss_aux_G_A = 0.0
        self.loss_aux_G_B = 0.0
        
        if self.opt.one_side:
            self.loss_G = self.loss_G_A + self.loss_cycle_A + self.loss_aux_G_A
            self.loss_G.backward()
            self.optimizer_G_A.step()
            self.loss_G = self.loss_G_B + self.loss_cycle_B + self.loss_aux_G_B
            self.loss_G.backward()
            self.optimizer_G_B.step()
        else:
            self.loss_G = self.loss_G_A + self.loss_cycle_A + self.loss_aux_G_A
            self.loss_G += self.loss_G_B + self.loss_cycle_B + self.loss_aux_G_B
            self.loss_G.backward()
            self.optimizer_G_B.step()
            self.optimizer_G_A.step()

    def optimize_parameters(self, total_iter=None):
        #if total_iter % 100 == 0:
        # forward
        self.forward()
        # G_A and G_B
        self.backward_G()
        if total_iter is not None and total_iter % self.opt.update_D != 0:
            return
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_lr(self):
        lr_D = self.optimizer_D_A.param_groups[0]['lr']
        lr_G = self.optimizer_G_A.param_groups[0]['lr']
        return OrderedDict([('D', lr_D), ('G', lr_G)])

    def get_current_errors(self):
        ret = OrderedDict([])
        for i in range(len(self.loss_D_A_real)):
            D_A_real = self.loss_D_A_real[i]
            D_A_fake = self.loss_D_A_fake[i]
            G_A = self.loss_G_A_list[i]
            D_B_real = self.loss_D_B_real[i]
            D_B_fake = self.loss_D_B_fake[i]
            G_B = self.loss_G_B_list[i]
            ret = OrderedDict(list(ret.items()) + [('D_A_real_%d'%i, D_A_real), ('D_A_fake_%d'%i, D_A_fake), ('G_A_%d'%i, G_A), ('D_B_real_%d'%i, D_B_real), ('D_B_fake_%d'%i, D_B_fake), ('G_B_%d'%i, G_B)])
            if self.opt.lambda_gp > 0.0:
                gp_A = self.loss_D_A_gp[i]
                gp_B = self.loss_D_B_gp[i]
                ret = OrderedDict(list(ret.items()) + [('D_A_gp_%d'%i, gp_A), ('D_B_gp_%d'%i, gp_B)])
        for i in range(len(self.loss_cycle_A_list)):
            Cyc_A = self.loss_cycle_A_list[i]
            Cyc_B = self.loss_cycle_B_list[i]
            ret = OrderedDict(list(ret.items()) + [('Cyc_A_%d'%i, Cyc_A), ('Cyc_B_%d'%i, Cyc_B)])
            if self.opt.identity > 0.0:
                idt_A = self.loss_idt_A.data[0]
                idt_B = self.loss_idt_B.data[0]
                ret = OrderedDict(list(ret.items()) + [('idt_A', idt_A), ('idt_B', idt_B)])
            if self.opt.alpha_gate != '' and i < len(self.loss_D_A_real) - 1:
                if self.opt.alpha_gate == 'simp':
                    mask_A = self.netG_A.gate.mask
                    mask_B = self.netG_B.gate.mask
                    ret = OrderedDict(list(ret.items()) + [('G_A_mask', mask_A), ('G_B_mask', mask_B)])
                elif 'trans' not in  self.opt.alpha_gate:
                    mask_A = eval('self.netG_A.gate%d.mask.mean()'%i)
                    mask_B = eval('self.netG_B.gate%d.mask.mean()'%i)
                    ret = OrderedDict(list(ret.items()) + [('G_A_mask_%d'%i, mask_A.data[0]), ('G_B_mask_%d'%i, mask_B.data[0])])
        if self.opt.lambda_color_mean > 0 or self.opt.lambda_color_sig_mean > 0:
            ret = OrderedDict(list(ret.items()) + [('G_A_color', self.loss_color_A.data[0]), ('G_B_color', self.loss_color_B.data[0])])
        if self.opt.log_grad:
            g_D_A = util.get_grads(self.netD_A, ret_type='sum').data[0]
            g_D_B = util.get_grads(self.netD_B, ret_type='sum').data[0]
            g_G_A = util.get_grads(self.netG_A, ret_type='sum').data[0]
            g_G_B = util.get_grads(self.netG_B, ret_type='sum').data[0]
            ret = OrderedDict(list(ret.items()) + [('D_A_grad', g_D_A), ('D_B_grad', g_D_B), ('G_A_grad', g_G_A), ('G_B_grad', g_G_B)])
        if self.opt.lambda_color_mean > 0 or self.opt.lambda_color_sig_mean > 0:
            ret = OrderedDict(list(ret.items()) + [('G_A_color', self.loss_color_A.data[0]), ('G_B_color', self.loss_color_B.data[0])])
        if self.opt.lambda_style > 0:
            ret = OrderedDict(list(ret.items()) + [('G_A_style', self.style_loss_A.data[0]), ('G_B_style', self.style_loss_B.data[0])])
        if self.opt.lambda_content > 0 or self.opt.lambda_content_l1 > 0:
            ret = OrderedDict(list(ret.items()) + [('G_A_cont', self.content_loss_A.data[0]), ('G_B_cont', self.content_loss_B.data[0])])
        return ret

    def pad_to_size(self, tensor):
        img = util.tensor2im(tensor)
        size = img.shape[1]
        pad_size = (self.opt.fineSize - size) / 2
        pad_shape = (pad_size, pad_size)
        return np.pad(img, (pad_shape, pad_shape, (0,0)), 'constant')
    
    def get_current_visuals(self):
        real_A = self.pad_to_size(self.real_A.data)
        ret = OrderedDict([('real_A', real_A)])
        if not isinstance(self.fake_A, (tuple, list)):
            self.fake_A = (self.fake_A, )
        if not isinstance(self.fake_B, (tuple, list)):
            self.fake_B = (self.fake_B, )
        if not isinstance(self.rec_A, (tuple, list)):
            self.rec_A = (self.rec_A, )
        if not isinstance(self.rec_B, (tuple, list)):
            self.rec_B = (self.rec_B, )
        for i in range(len(self.fake_A)):
            fake_B = self.pad_to_size(self.fake_B[i].data)
            rec_A = self.pad_to_size(self.rec_A[i].data)
            if self.opt.alpha_gate != '' and 'simp' not in self.opt.alpha_gate and i < len(self.fake_A) - 1 and 'trans' not in  self.opt.alpha_gate:
                mask_A = self.pad_to_size(eval('self.netG_B.gate%d.mask.data'%i))
                ret = OrderedDict(list(ret.items()) + [('mask_A_%d'%i, mask_A)])
            ret = OrderedDict(list(ret.items()) + [('fake_B_%d'%i, fake_B), ('rec_A_%d'%i, rec_A)])
        real_B = self.pad_to_size(self.real_B.data)
        ret = OrderedDict(list(ret.items()) + [('real_B', real_B)])
        for i in range(len(self.fake_A)):
            fake_A = self.pad_to_size(self.fake_A[i].data)
            rec_B = self.pad_to_size(self.rec_B[i].data)
            if self.opt.alpha_gate != '' and 'simp' not in self.opt.alpha_gate and i < len(self.fake_A) - 1 and 'trans' not in  self.opt.alpha_gate:
                mask_B = self.pad_to_size(eval('self.netG_A.gate%d.mask.data'%i))
                ret = OrderedDict(list(ret.items()) + [('mask_B_%d'%i, mask_B)])
            ret = OrderedDict(list(ret.items()) + [('fake_A_%d'%i, fake_A), ('rec_B_%d'%i, rec_B)])
        del self.fake_A, self.fake_B, self.rec_A, self.rec_B
        return ret

    def get_mse_error(self):
        # assuming (A, B) is exact pair
        ret = []
        for i in range(len(self.fake_A)):
            np_fake_A = util.tensor2im(self.fake_A[i].data)
            np_fake_B = util.tensor2im(self.fake_B[i].data)
            np_A = util.tensor2im(self.real_A.data)
            np_B = util.tensor2im(self.real_B.data)
            np_A = imresize(np_A, np_fake_A.shape[:2])
            np_B = imresize(np_B, np_fake_B.shape[:2])
            ret.append((((np_A - np_fake_A)**2).sum()/np_fake_A.size, ((np_B - np_fake_B)**2).sum()/np_fake_B.size))
        return ret

    def get_network_params(self):
        return []

    def get_network_grads(self):
        return []

    def save(self, label):
        CycleGANModel.save(self, label)
    
    def eval_network(self):
        dataset = self.eval_data_loader.load_data()
        sum_mse_A, sum_mse_B = [], []
        for i in range(self.opt.num_D):
            sum_mse_A.append(0)
            sum_mse_B.append(0)
        ret = OrderedDict([])
        for data in tqdm.tqdm(dataset):
            self.set_input(data)
            self.test()
            mse = self.get_mse_error()
            for idx, i in enumerate(mse):
                sum_mse_A[idx] += i[0]
                sum_mse_B[idx] += i[1]
        for i in range(self.opt.num_D):
            ret = OrderedDict(list(ret.items()) + [( 'G_B_%d'%i, sum_mse_A[i] / float(len(self.eval_data_loader)) ), ( 'G_A_%d'%i, sum_mse_B[i] / float(len(self.eval_data_loader)))])
        return ret

