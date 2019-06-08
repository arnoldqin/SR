from __future__ import absolute_import
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import utils.util as util
from utils.image_pool import ImagePool
from .base_model import BaseModel
from models import networks
import sys
from utils.util import load_network_with_path, accumulate, set_eval
from scipy.misc import imresize
import tqdm


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.target_weight = []
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        if opt.aux:
                self.A_aux = self.Tensor(nb, opt.input_nc, size, size)
                self.B_aux = self.Tensor(nb, opt.output_nc, size, size)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, opt=opt)
        self.netG_A_running = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, opt=opt)
        set_eval(self.netG_A_running)
        accumulate(self.netG_A_running, self.netG_A, 0)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, opt=opt)
        self.netG_B_running = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, opt=opt)
        set_eval(self.netG_B_running)
        accumulate(self.netG_B_running, self.netG_B, 0)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt=opt)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt=opt)
        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A, opt, (opt.input_nc, opt.fineSize, opt.fineSize))
        # networks.print_network(self.netG_B, opt)
        if self.isTrain:
            networks.print_network(self.netD_A, opt)
            # networks.print_network(self.netD_B, opt)
        print('-----------------------------------------------')


        if not self.isTrain or opt.continue_train:
            print('Loaded model')
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netG_A_running, 'G_A', which_epoch)
                self.load_network(self.netG_B_running, 'G_B', which_epoch)
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain and opt.load_path != '':
            print('Loaded model from load_path')
            which_epoch = opt.which_epoch
            load_network_with_path(self.netG_A, 'G_A', opt.load_path, epoch_label=which_epoch)
            load_network_with_path(self.netG_B, 'G_B', opt.load_path, epoch_label=which_epoch)
            load_network_with_path(self.netD_A, 'D_A', opt.load_path, epoch_label=which_epoch)
            load_network_with_path(self.netD_B, 'D_B', opt.load_path, epoch_label=which_epoch)
                
        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            if len(self.target_weight) == opt.num_D: 
                print(self.target_weight)
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor, target_weight=self.target_weight, gan=opt.gan)
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor, gan=opt.gan)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionColor = networks.ColorLoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = (input['A_paths'], input['B_paths'])
        if self.opt.aux:
                input_A_aux = input['A_aux' if AtoB else 'B_aux']
                input_B_aux = input['B_aux' if AtoB else 'A_aux']
                self.A_aux.resize_(input_A_aux.size()).copy_(input_A_aux)
                self.B_aux.resize_(input_B_aux.size()).copy_(input_B_aux)

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG_A.forward(self.real_A)
        self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG_B.forward(self.real_B)
        self.rec_B = self.netG_A.forward(self.fake_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True, self.opt.lambda_adv)
        # Fake
        if isinstance(fake, (tuple, list)):
            detach_fake = [i.detach() for i in fake]
        else:
            detach_fake = fake.detach()
        pred_fake = netD.forward(detach_fake)
        loss_D_fake = self.criterionGAN(pred_fake, False, self.opt.lambda_adv)

        if isinstance(loss_D_real, (tuple, list)):
            ret = (loss_D_real[-1], loss_D_fake[-1])
            loss_D_real = loss_D_real[0]
            loss_D_fake = loss_D_fake[0]
        else:
            ret = (loss_D_real, loss_D_fake)
        if self.opt.lambda_gp > 0:
            # Gradient Penalty
            alpha = torch.rand(self.opt.batchSize, 1, 1, 1).expand(real.size()).cuda()
            if self.opt.gp == 'dragan':
                x_hat = Variable(alpha * real.data + (1 - alpha) *
                        (real.data + 0.5 * real.data.std() * torch.rand(real.size()).cuda()), requires_grad=True)
            elif self.opt.gp == 'wgangp':
                x_hat = Variable(alpha * real.data + (1 - alpha) *  detach_fake.data, requires_grad=True)
            else:
                x_hat = Variable(alpha * detach_fake.data + (1 - alpha) *
                        (detach_fake.data + 0.5 * detach_fake.data.std() * torch.rand(detach_fake.size()).cuda()), requires_grad=True)
            pred_hat = netD.forward(x_hat)
            if isinstance(pred_hat,(tuple, list)):
                gradient_penalty = 0.0
                gradient_penalty_list = []
                for i in range(len(pred_hat)):
                    gradients = torch.autograd.grad(outputs=pred_hat[i][-1], inputs=x_hat, grad_outputs=torch.ones(pred_hat[i][-1].size()).cuda(), create_graph=True, retain_graph=True, only_inputs=True)[0]
                    if self.opt.weight_adv is not None:
                        current_gradient_penalty = self.weight_adv[i] * ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.opt.lambda_gp
                    else:
                        current_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                    gradient_penalty_list.append(current_gradient_penalty.data[0])
                    gradient_penalty += current_gradient_penalty
                ret += (gradient_penalty_list,)
            else:
                gradients = torch.autograd.grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(), create_graph=True, retain_graph=True, only_inputs=True)[0]
                gradient_penalty = self.opt.lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                ret += (gradient_penalty,)
            loss_D = (loss_D_real + loss_D_fake) + gradient_penalty
        else:
            gradient_penalty = 0.0
            loss_D = (loss_D_real + loss_D_fake) * 0.5

        loss_D.backward(retain_graph=True)
        return ret

    def backward_D_A(self):
        if self.opt.eval_to_dis:
            set_eval(self.netG_A)
            self.fake_B = self.netG_A.forward(self.real_A)
            self.netG_A.train()
        fake_B = self.fake_B_pool.query(self.fake_B)
        if self.opt.lambda_gp > 0:
            self.loss_D_A_real, self.loss_D_A_fake, self.loss_D_A_gp = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        else:
            self.loss_D_A_real, self.loss_D_A_fake = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
            self.loss_D_A_gp = 0.0

    def backward_D_B(self):
        if self.opt.eval_to_dis:
            set_eval(self.netG_B)
            self.fake_A = self.netG_B.forward(self.real_B)
            self.netG_B.train()
        fake_A = self.fake_A_pool.query(self.fake_A)
        if self.opt.lambda_gp > 0:
            self.loss_D_B_real, self.loss_D_B_fake, self.loss_D_B_gp = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        else:
            self.loss_D_B_real, self.loss_D_B_fake = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
            self.loss_D_B_gp = 0.0

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_rec = self.opt.lambda_rec
        lambda_adv = self.opt.lambda_adv
        lambda_color = self.opt.lambda_color_mean

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A.forward(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_rec * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B.forward(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_rec * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        # D_A(G_A(A))
        self.fake_B = self.netG_A.forward(self.real_A)
        pred_fake = self.netD_A.forward(self.fake_B)
        self.loss_G_A = self.criterionGAN(pred_fake, True) * lambda_adv
        # D_B(G_B(B))
        self.fake_A = self.netG_B.forward(self.real_B)
        pred_fake = self.netD_B.forward(self.fake_A)
        self.loss_G_B = self.criterionGAN(pred_fake, True) * lambda_adv
        # Forward cycle loss
        # if self.opt.eval_to_rec: TODO?
            # pass
        self.rec_A = self.netG_B.forward(self.fake_B)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_rec
        # Backward cycle loss
        self.rec_B = self.netG_A.forward(self.fake_A)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_rec
        # aux loss
        self.loss_aux_G_A = 0.0
        self.loss_aux_G_B = 0.0
        if self.opt.aux:
                pass_B = not ('B' in self.opt.aux_loss)
                pass_A = not ('A' in self.opt.aux_loss)
                self.loss_color_B = self.criterionColor(self.A_aux, self.fake_B, pass_B or (not self.opt.lambda_color_mean > 0))
                self.loss_aux_G_B += self.loss_color_B 
                self.loss_color_A = self.criterionColor(self.B_aux, self.fake_A, pass_A or (not self.opt.lambda_color_mean > 0))
                self.loss_aux_G_A += self.loss_color_A

        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G += (self.loss_aux_G_A + self.loss_aux_G_B) * lambda_color
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        accumulate(self.netG_A_running, self.netG_A)
        accumulate(self.netG_B_running, self.netG_B)
        # if total_iter is not None and total_iter % self.opt.update_D != 0:
        #     return
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        D_A_real = self.loss_D_A_real.item()
        D_A_fake = self.loss_D_A_fake.item()
        G_A = self.loss_G_A.item()
        G_A = self.loss_G_A.item()
        Cyc_A = self.loss_cycle_A.item()
        D_B_real = self.loss_D_B_real.item()
        D_B_fake = self.loss_D_B_fake.item()
        G_B = self.loss_G_B.item()
        Cyc_B = self.loss_cycle_B.item()
        ret = OrderedDict([('D_A_real', D_A_real), ('D_A_fake', D_A_fake), ('G_A', G_A), ('Cyc_A', Cyc_A),
                           ('D_B_real', D_B_real), ('D_B_fake', D_B_fake), ('G_B', G_B), ('Cyc_B', Cyc_B)])
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.item()
            idt_B = self.loss_idt_B.item()
            ret = OrderedDict(list(ret.items()) + [('idt_A', idt_A), ('idt_B', idt_B)])
        if self.opt.lambda_gp > 0.0:
            gp_A = self.loss_D_A_gp.item()
            gp_B = self.loss_D_B_gp.item()
            ret = OrderedDict(list(ret.items()) + [('D_A_gp', gp_A), ('D_B_gp', gp_B)])
        if self.opt.lambda_color_mean > 0 or self.opt.lambda_color_sig_mean > 0:
            ret = OrderedDict(list(ret.items()) + [('G_A_color', self.loss_color_A.data[0]), ('G_B_color', self.loss_color_B.data[0])])
        if self.opt.log_grad:
            g_D_A = util.get_grads(self.netD_A, ret_type='sum').item()
            g_D_B = util.get_grads(self.netD_B, ret_type='sum').item()
            g_G_A = util.get_grads(self.netG_A, ret_type='sum').item()
            g_G_B = util.get_grads(self.netG_B, ret_type='sum').item()
            ret = OrderedDict(list(ret.items()) + [('D_A_grad', g_D_A), ('D_B_grad', g_D_B), ('G_A_grad', g_G_A), ('G_B_grad', g_G_B)])

        return ret

    def get_mse_error(self):
        # assuming (A, B) is exact pair
        np_A = util.tensor2im(self.real_A.data)
        np_B = util.tensor2im(self.real_B.data)
        np_fake_A = util.tensor2im(self.fake_A.data)
        np_fake_B = util.tensor2im(self.fake_B.data)
        return ((np_A - np_fake_A)**2).sum()/np_fake_A.size, ((np_B - np_fake_B)**2).sum()/np_fake_B.size
    
    def get_current_lr(self):
        lr_A = self.optimizer_D_A.param_groups[0]['lr']
        lr_B = self.optimizer_D_B.param_groups[0]['lr']
        lr_G = self.optimizer_G.param_groups[0]['lr']
        return OrderedDict([('D_A', lr_A), ('D_B', lr_B), ('G', lr_G)])

    def get_current_visuals(self):
        running_fake_B = self.netG_A_running.forward(self.real_A)
        running_fake_A = self.netG_B_running.forward(self.real_B)
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        running_fake_B_img = util.tensor2im(running_fake_B.data)
        rec_A = util.tensor2im(self.rec_A.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)
        running_fake_A_img = util.tensor2im(running_fake_A.data)
        rec_B = util.tensor2im(self.rec_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('running_B', running_fake_B_img), ('rec_A', rec_A),
                            ('real_B', real_B), ('fake_A', fake_A), ('running_A', running_fake_A_img), ('rec_B', rec_B)])

    def get_network_params(self):
        return [('G_A', util.get_params(self.netG_A)),
                ('G_B', util.get_params(self.netG_B)),
                ('D_A', util.get_params(self.netD_A)),
                ('D_B', util.get_params(self.netD_B))]

    def get_network_grads(self):
        return [('G_A', util.get_grads(self.netG_A)),
                ('G_B', util.get_grads(self.netG_B)),
                ('D_A', util.get_grads(self.netD_A)),
                ('D_B', util.get_grads(self.netD_B))]

    def save(self, label):
        self.save_network(self.netG_A_running, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B_running, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

    def eval_network(self):
        dataset = self.eval_data_loader.load_data()
        sum_mse_A, sum_mse_B = 0, 0
        for data in tqdm.tqdm(dataset):
            self.set_input(data)
            self.test()
            mse_A, mse_B = self.get_mse_error()
            sum_mse_A += mse_A
            sum_mse_B += mse_B
        return OrderedDict([( 'G_B', sum_mse_A / float(len(self.eval_data_loader)) ), ( 'G_A', sum_mse_B / float(len(self.eval_data_loader)) ) ])
