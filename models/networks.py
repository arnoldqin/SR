import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
from collections import namedtuple
import utils.util as util
import modules.architecture as arch

# {{{ <editor-fold desc="Functions">
def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal',
             gpu_ids=[], n_downsampling=2, n_upsampling=2, n_resblocks=None, side='A', opt=None, mult=None):
    netG = None
    n_resblocks = opt.n_resblocks if n_resblocks is None else n_resblocks
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                               n_blocks=n_resblocks, gpu_ids=gpu_ids, n_downsampling=opt.n_downsample,
                               n_upsampling=opt.n_upsample, opt=opt)
    elif which_model_netG == 'resnet_stackv2':
        netG = ResnetGeneratorStackv2(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                      n_blocks=n_resblocks, gpu_ids=gpu_ids, n_downsampling=opt.n_downsample,
                                      n_upsampling=opt.n_upsample, opt=opt)
    elif which_model_netG == 'resnet_noise':
        netG = ResnetGeneratorNoise(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                    n_blocks=n_resblocks, gpu_ids=gpu_ids, n_downsampling=opt.n_downsample,
                                    n_upsampling=opt.n_upsample, opt=opt)
    elif which_model_netG == 'resnet_9blocks_old':
        netG = ResnetGeneratorLegacy(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                     n_blocks=9, gpu_ids=gpu_ids, opt=opt)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6,
                               gpu_ids=gpu_ids, opt=opt)
    elif which_model_netG == 'resnet_orig':
        netG = ResnetGeneratorOrig(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6,
                                   gpu_ids=gpu_ids, opt=opt)
    elif which_model_netG == 'resnet_n_blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                               n_blocks=n_resblocks, gpu_ids=gpu_ids, n_downsampling=n_downsampling,
                               n_upsampling=n_upsampling, opt=opt)
    elif which_model_netG == 'unet_64':
        netG = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             gpu_ids=gpu_ids)
    elif which_model_netG == 'gen':
        netG = Generator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3,
                         n_downsampling=n_downsampling, n_upsampling=n_upsampling, gpu_ids=gpu_ids, opt=opt)
    elif which_model_netG == 'gen_s2':
        netG = Generator_stage2only(norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids, opt=opt)
    elif which_model_netG == 'core':
        netG = Generator_core(input_nc, output_nc, mid_nc=opt.pre.ngf, ngf=opt.ngf, n_blocks=n_resblocks,
                              norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids,
                              n_downsampling=n_downsampling, n_upsampling=n_upsampling, opt=opt)
    elif which_model_netG == 'end2end':
        netG = Generator_end2end(opt=opt)
    elif which_model_netG == 'stack':
        netG = Generator_stack(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                               n_blocks=n_resblocks, gpu_ids=gpu_ids, n_downsampling=opt.n_downsample,
                               n_upsampling=opt.n_upsample, side=side, opt=opt)
    elif which_model_netG == 'stack_simp':
        netG = Generator_stack_simp(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                    n_blocks=n_resblocks, gpu_ids=gpu_ids, n_downsampling=opt.n_downsample,
                                    n_upsampling=opt.n_upsample, side=side, opt=opt)
    elif which_model_netG == 'hourglass':
        netG = Hourglass_residual(n_resblocks, input_nc)
    elif which_model_netG == 'ResnetEncoder_my':
        netG = ResnetEncoder_my(input_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                               n_blocks=n_resblocks, gpu_ids=gpu_ids, n_downsampling=n_downsampling,
                               opt=opt)
    elif which_model_netG == 'ResnetDecoder_my':
        netG = ResnetDecoder_my(mult,  ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                               n_blocks=n_resblocks, gpu_ids=gpu_ids,
                               n_upsampling=n_upsampling, opt=opt)
    elif which_model_netG == 'GeneratorLL':
        netG = GeneratorLL(mult,  output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                               n_blocks=n_resblocks, gpu_ids=gpu_ids, opt=opt)


    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    if 'stack' in which_model_netG and opt.load_pre:
        print('loaded pre-network from', opt.pre_path)
        util.load_network_with_path(netG.sub_model, 'G_' + side, opt.pre_path)
    return netG


def define_D(input_nc, ndf, which_model_netD, n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal',
             gpu_ids=[], one_out=None, feat_len=None, opt=None):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids,
                                   one_out=one_out, feat_len=feat_len, opt=opt)
    elif which_model_netD == 'n_layers_attr':
        netD = NLayerAttrDiscriminator(input_nc, 1, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids,
                                   one_out=one_out, feat_len=feat_len, opt=opt)
    elif which_model_netD == 'multiscale':
        netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                       num_D=opt.num_D, opt=opt)
    elif which_model_netD == 'n_layers_stackv2':
        netD = MultiDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                  num_D=opt.num_D, opt=opt)
    elif which_model_netD == 'dis':
        netD = Discriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, gpu_ids=gpu_ids, opt=opt)
    elif which_model_netD == 'core':
        netD = Discriminator_core(opt.pre.ndf, ndf, n_layers=opt.n_layers_D, norm_layer=norm_layer, opt=opt)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    init_weights(netD, init_type=init_type)
    return netD


def print_network_shape(model, input_size):
    class TablePrinter(object):
        def __init__(self, fmt, sep=' ', ul=None):
            super(TablePrinter, self).__init__()
            self.fmt = str(sep).join(
                '{lb}{0}:{1}{rb}'.format(key, width, lb='{', rb='}') for heading, key, width in fmt)
            self.head = {key: heading for heading, key, width in fmt}
            self.ul = {key: str(ul) * width for heading, key, width in fmt} if ul else None
            self.width = {key: width for heading, key, width in fmt}

        def row(self, data):
            return self.fmt.format(**{k: str(data.get(k, ''))[:w] for k, w in self.width.items()})

        def __call__(self, dataList):
            _r = self.row
            res = [_r(data) for data in dataList]
            res.insert(0, _r(self.head))
            if self.ul:
                res.insert(1, _r(self.ul))
            return '\n'.join(res)

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            idx = module_idx + 1
            m_key = '%i-%s' % (idx, class_name)
            tmp = {}
            tmp['name'] = m_key
            tmp['input_shape'] = list(input[0].size())
            tmp['output_shape'] = list(output.size())
            tmp['nb_params'] = 0
            for param in module.parameters():
                tmp['nb_params'] += param.numel()
            summary.append(tmp)

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_hook(hook))

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1, *in_size), volatile=True).cuda() for in_size in input_size]
    else:
        x = Variable(torch.rand(1, *input_size), volatile=True).cuda()

    # create properties
    summary = []
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    if isinstance(input_size[0], (list, tuple)):
        y = model(*x)
    else:
        y = model(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    fmt = [('Name', 'name', 30), ('In Shape', 'input_shape', 25), ('Out Shape', 'output_shape', 25),
           ('Params', 'nb_params', 15)]
    print(TablePrinter(fmt, ul='=')(summary))
    return summary

def print_network(net, opt, input_shape=None):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    if input_shape is None:
        print_network_shape(net, (opt.input_nc, opt.fineSize, opt.fineSize))
    else:
        print_network_shape(net, input_shape)


# }}} </editor-fold>

# {{{ <editor-fold desc="Self-define Layers">
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class GaussianNoise(nn.Module):
    def __init__(self, stddev=0.2):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
        return din


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class AlphaGateSimp(nn.Module):

    def __init__(self, gate_type=''):
        super(AlphaGateSimp, self).__init__()
        self.gate_simp = (gate_type == 'simp')
        if self.gate_simp:
            self.mask = 0.0
        else:
            self.alpha = nn.Parameter(torch.ones(1) * -3.0)
            self.th = 0.85
            self.passmode = False

    def forward(self, x_0, x_1, t):
        if self.gate_simp:
            return self.mask * x_1 + (1 - self.mask) * x_0
        else:
            self.mask = F.sigmoid(self.alpha)
            # if not self.passmode and (self.mask < self.th).all():
            return self.mask * x_1 + (1 - self.mask) * x_0
        # else:
        # self.passmode = True
        # return x_1


class AlphaGate(nn.Module):
    def __init__(self, in_nc, not_mono_gate=False, gate_type=''):
        super(AlphaGate, self).__init__()
        activation = nn.ReLU(True)
        use_bias = True
        norm_layer = nn.InstanceNorm2d
        out_nc = 3 if not_mono_gate else 1
        mid_nc = 64
        self.mask = None
        self.gate_type = gate_type
        self.pass_gate = False
        if 'trans' not in gate_type:
            model = [nn.Conv2d(in_nc, mid_nc, kernel_size=3, padding=1, bias=use_bias),
                     norm_layer(mid_nc),
                     activation,
                     nn.Conv2d(mid_nc, mid_nc, kernel_size=3, padding=1, bias=use_bias),
                     norm_layer(mid_nc),
                     activation,
                     nn.Conv2d(mid_nc, out_nc, kernel_size=3, padding=1, bias=use_bias),
                     nn.Sigmoid()]
            self.model = nn.Sequential(*model)
        else:
            out_nc = 3
            model = [norm_layer(in_nc),
                     activation,
                     nn.Conv2d(in_nc, out_nc, kernel_size=1, padding=0, bias=False)]
            self.model = nn.Sequential(*model)

    def forward(self, x_0, x_1, t):
        if not self.pass_gate:
            if self.gate_type == '0_1':
                self.mask = self.model(torch.cat((t, x_0), 1))
            elif self.gate_type == '0_1_2':
                self.mask = self.model(torch.cat((t, x_0, x_1), 1))
            elif self.gate_type == '1_2':
                self.mask = self.model(torch.cat((x_0, x_1), 1))
            elif self.gate_type == 'trans_2':
                return self.model(x_1)
            elif self.gate_type == 'trans_1_2':
                return self.model(torch.cat((x_0, x_1), 1))
            # if (self.mask.mean() > 0.8).all():
            # self.pass_gate = True
            return self.mask * x_1 + (1 - self.mask) * x_0
        else:
            return x_1


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, shallow=False, use_lrelu=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, shallow,
                                                use_lrelu)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, shallow, use_lrelu):
        conv_block = []
        if use_lrelu:
            activation = nn.LeakyReLU(0.2, True)
        else:
            activation = nn.ReLU(True)
        if not shallow:
            p = 0
            if padding_type == 'reflect':
                conv_block += [nn.ReflectionPad2d(1)]
            elif padding_type == 'replicate':
                conv_block += [nn.ReplicationPad2d(1)]
            elif padding_type == 'zero':
                p = 1
            else:
                raise NotImplementedError('padding [%s] is not implemented' % padding_type)

            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                           norm_layer(dim),
                           activation]
            if use_dropout:
                conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Define a downsample block
class DownBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, use_noise):
        super(DownBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, use_noise)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, use_noise):
        conv_block = []

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=p, bias=use_bias),
                       norm_layer(dim)]

        if use_noise:
            conv_block += [GaussianNoise()]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReLU(True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        return out


class NNBlock(nn.Module):
    def __init__(self, ch0, ch1, nn_type='conv', kw=3, norm=nn.BatchNorm2d, activation=None, dropout=False, noise=False,
                 pre_activation=False):
        super(NNBlock, self).__init__()
        self.dropout = dropout
        self.nn_type = nn_type

        if nn_type == 'down_conv':
            c = [nn.Conv2d(ch0, ch1, kernel_size=3, stride=2, padding=1)]
        elif nn_type == 'up_conv':
            c = [nn.ConvTranspose2d(ch0, ch1, kernel_size=3, stride=2, padding=1, output_padding=1)]
        elif nn_type == 'up_conv4':
            c = [nn.ConvTranspose2d(ch0, ch1, kernel_size=4, stride=2, padding=1)]
        elif nn_type == 'up_subpix':
            pw = kw // 2
            c = [nn.Conv2d(ch0, ch1 * 4, kernel_size=kw, stride=1, padding=pw)]
        elif nn_type == 'conv' or nn_type == 'up_scale':
            pw = kw // 2
            c = [nn.Conv2d(ch0, ch1, kernel_size=kw, stride=1, padding=pw)]
        elif nn_type == 'linear':
            c = [nn.Linear(ch0, ch1)]
        elif nn_type == 'down_conv7':
            c = [nn.ReflectionPad2d(3),
                 nn.Conv2d(ch0, ch1, kernel_size=7, padding=0), ]
        else:
            raise NotImplementedError('layer [%s] is not found' % nn_type)

        seq = []
        if activation is not None and pre_activation:
            seq += [activation, ]
        if nn_type == 'up_scale':
            seq += [nn.Upsample(scale_factor=2, mode='nearest'), ]
        seq += c
        if nn_type == 'up_subpix':
            seq += [nn.PixelShuffle(2), ]
        if norm is not None:
            seq += [norm(ch1), ]
        if noise:
            seq += [GaussianNoise(), ]
        if dropout:
            seq += [nn.Dropout(0.5), ]
        if activation is not None and not pre_activation:
            seq += [activation, ]

        self.model = nn.Sequential(*seq)

    def forward(self, input):
        return self.model(input)


class ResBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.LeakyReLU(0.2, True), use_dropout=False, use_bias=True,
                 shallow=False, pre_activation=False):
        super(ResBlock, self).__init__()
        conv_block = []
        if pre_activation and activation is not None:
            conv_block += [activation, ]
        if not shallow:
            conv_block += [nn.ReflectionPad2d(1),
                           nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), ]
            if norm_layer is not None:
                conv_block += [norm_layer(dim), ]
            conv_block += [activation, ]
            if use_dropout:
                conv_block += [nn.Dropout(0.5), ]
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), ]
        if norm_layer is not None:
            conv_block += [norm_layer(dim), ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class DownResBlock(nn.Module):
    def __init__(self, ch0, ch1, norm=None, activation=None, dropout=False, noise=False):
        super(DownResBlock, self).__init__()
        conv_block = [ResBlock(ch0, norm_layer=norm, activation=activation, use_dropout=dropout, pre_activation=True),
                      ResBlock(ch0, norm_layer=norm, activation=activation, use_dropout=dropout, pre_activation=True),
                      NNBlock(ch0, ch1, nn_type='down_conv', norm=norm, activation=activation, dropout=dropout,
                              noise=noise, pre_activation=True)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        return out


# }}} </editor-fold>

# {{{ <editor-fold desc="LOSS Layers">
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, target_weight=None, gan='lsgan'):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.target_weight = target_weight
        if gan == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan == 'wgan':
            self.fake_label = 1.0
            self.real_label = -1.0
            self.loss = lambda x, l: (x * l).mean()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real, lambda_adv=1.0):
        if isinstance(input, (list, tuple)):
            loss = 0
            loss_list = []
            for i, input_i in enumerate(input):
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                if self.target_weight is None:
                    current_loss = self.loss(pred, target_tensor) * lambda_adv
                    loss_list.append(current_loss.data[0])
                    loss += current_loss
                else:
                    current_loss = self.target_weight[i] * self.loss(pred, target_tensor) * lambda_adv
                    loss_list.append(current_loss.data[0])
                    loss += current_loss
            return loss, loss_list
        else:
            # import ipdb; ipdb.set_trace()
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class RECLoss(nn.Module):
    def __init__(self, tensor=torch.FloatTensor, rec='l1', target_weight=None):
        super(RECLoss, self).__init__()
        self.Tensor = tensor
        self.target_weight = target_weight
        if rec == 'l1':
            self.loss = nn.L1Loss()
        else:
            raise (NotImplementedError)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def __call__(self, input, target, lambda_rec=1.0):
        if isinstance(input, (list, tuple)):
            loss = 0
            target_tensor = target
            loss_list = []
            for i, input_i in enumerate(input):
                pred = input_i[-1]
                if self.target_weight is None:
                    current_loss = self.loss(pred, target_tensor) * lambda_rec
                    loss_list.append(current_loss.data[0])
                    loss += current_loss
                else:
                    current_loss = self.target_weight[i] * self.loss(pred, target_tensor) * lambda_rec
                    loss_list.append(current_loss.data[0])
                    loss += current_loss
                target_tensor = self.downsample(target_tensor)
            return loss, loss_list
        else:
            return self.loss(input[-1], target)


class VGGLossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(VGGLossNetwork, self).__init__()
        self.vgg_layers = vgg_model.features
        self.LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3",
            '29': "relu5_3"
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return self.LossOutput(**output)


class ColorLoss(nn.Module):
    def __init__(self, lambda_color_mean=1.0, lambda_color_var=5.0, use_sig=False, tensor=torch.FloatTensor):
        super(ColorLoss, self).__init__()
        self.lambda_color_mean = lambda_color_mean
        self.lambda_color_var = lambda_color_var
        self.Tensor = tensor
        self.use_sig = use_sig

    def __call__(self, input, target, pass_loss=False):
        if pass_loss:
            return Variable(torch.zeros(1).cuda())
        self.loss_color = 0.0
        if not self.use_sig:
            for i in range(3):
                self.loss_color += self.lambda_color_mean * (input[:, i, :].mean() - target[:, i, :].mean()) ** 2 + \
                                   self.lambda_color_var * (input[:, i, :].var() - target[:, i, :].var()) ** 2
        else:
            self.loss_color += self.opt.lambda_color_mean * (input.mean() - target.mean()) ** 2 + \
                               self.lambda_color_var * (input.var() - target.var()) ** 2
        return self.loss_color


class PerceptualLoss(nn.Module):
    def __init__(self, loss_net, lambda_style=5.0, lambda_content=1.0, use_l1=False, tensor=torch.FloatTensor):
        super(PerceptualLoss, self).__init__()
        self.lambda_style = lambda_style
        self.lambda_content = lambda_content
        self.Tensor = tensor
        self.loss_network = loss_net
        if use_l1:
            self.loss = util.mse_loss
        else:
            self.loss = util.l1_loss

    def gram_matrix(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    def trans_norm(self, x):
        rec = x * 0.5 + 0.5
        mean = Variable(
            torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)).astype('f')).cuda().expand_as(x))
        std = Variable(
            torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)).astype('f')).cuda().expand_as(x))
        return (rec - mean) / std

    def __call__(self, input, target, style_layers=4, cont_layer=1, pass_loss=False, lambda_style=None,
                 lambda_content=None):
        if pass_loss:
            return Variable(torch.zeros(1).cuda())
        assert (style_layers <= 4)
        assert (cont_layer <= 5)
        lambda_style = self.lambda_style if lambda_style is None else lambda_style
        lambda_content = self.lambda_content if lambda_content is None else lambda_content
        self.style_loss = 0.0
        self.content_loss = 0.0
        features_x = self.loss_network(self.trans_norm(input))
        features_y = self.loss_network(self.trans_norm(target))
        for m in range(style_layers):
            gram_s = self.gram_matrix(features_x[m])
            gram_y = self.gram_matrix(features_y[m])
            self.style_loss += lambda_style * self.loss(gram_y, gram_s)
        self.content_loss = lambda_content * self.loss(features_x[cont_layer - 1], features_y[cont_layer - 1])
        return self.content_loss + self.style_loss


# }}} </editor-fold>

# {{{ <editor-fold desc="Networks">
# Legacy ResnetGen, from Distance GAN
class ResnetGeneratorLegacy(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[]):
        assert (n_blocks >= 0)
        super(ResnetGeneratorLegacy, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
                 norm_layer(ngf, affine=True),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(ngf * mult * 2, affine=True),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlockLegacy(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2), affine=True),
                      nn.ReLU(True)]

        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)




# Legacy ResBlock, from Distance GAN
class ResnetBlockLegacy(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlockLegacy, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        # TODO: support padding types
        assert (padding_type == 'zero')
        p = 1

        # TODO: InstanceNorm
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


def ResnetEncoder(input_nc, ngf, n_downsampling, use_bias, norm_layer):
    model = [nn.ReflectionPad2d(3),
             nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                       bias=use_bias),
             norm_layer(ngf),
             nn.ReLU(True)]

    for i in range(n_downsampling):
        mult = 2 ** i
        model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                            stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * mult * 2),
                  nn.ReLU(True)]
    return model  # 2**(n_downsampling - 1) channels


def ResBlocks(ngf, n_downsampling, n_blocks, padding_type, use_bias, norm_layer, use_dropout):
    model = []
    mult = 2 ** n_downsampling
    for i in range(n_blocks):
        model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                              use_bias=use_bias)]
    return model


def ResNetDecoder(output_nc, ngf, n_downsampling, use_bias, norm_layer):
    model = []
    for i in range(n_downsampling):
        mult = 2 ** (n_downsampling - i)
        model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                     kernel_size=3, stride=2,
                                     padding=1, output_padding=1,
                                     bias=use_bias),
                  norm_layer(int(ngf * mult / 2)),
                  nn.ReLU(True)]
    model += [nn.ReflectionPad2d(3)]
    model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
    model += [nn.Tanh()]
    return model


class ResnetGeneratorOrig(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', n_downsampling=2, n_upsampling=2, opt=None):
        assert (n_blocks >= 0)
        super(ResnetGeneratorOrig, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        encoder = ResnetEncoder(input_nc, ngf, n_downsampling, use_bias, norm_layer)
        resblocks = ResBlocks(ngf, n_downsampling, n_blocks, padding_type, use_bias, norm_layer, use_dropout)
        decoder = ResNetDecoder(output_nc, ngf, n_downsampling, use_bias, norm_layer)

        self.encoder = nn.Sequential(*encoder)
        self.resblocks = nn.Sequential(*resblocks)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, input):
        y = self.encoder(input)
        y = self.resblocks(y)
        return self.decoder(y)


class ResnetGeneratorStackv2(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', n_downsampling=2, n_upsampling=2, opt=None):
        assert (n_blocks >= 0)
        assert (opt.num_D >= 1)
        self.num_G = opt.num_D
        self.opt = opt
        super(ResnetGeneratorStackv2, self).__init__()
        netG = ResnetGeneratorOrig(input_nc, output_nc, ngf, norm_layer=norm_layer, \
                                   use_dropout=use_dropout, n_blocks=opt.n_resblocks, gpu_ids=gpu_ids,
                                   n_downsampling=opt.n_downsample, n_upsampling=opt.n_upsample, opt=opt)
        setattr(self, 'net' + str(self.num_G - 1), netG)
        for i in range(1, self.num_G):
            if self.opt.dense:
                input_nc_new = input_nc * (1 + i)
            else:
                input_nc_new = input_nc * 2
            if 'trans' in self.opt.alpha_gate:
                output_nc_new = input_nc_new
            else:
                output_nc_new = output_nc
            if opt.unet:
                netG = UnetGenerator(input_nc_new, output_nc_new, opt.n_downsample, ngf, norm_layer=norm_layer,
                                     use_dropout=use_dropout, gpu_ids=gpu_ids)
            else:
                netG = ResnetGeneratorOrig(input_nc_new, output_nc_new, ngf, norm_layer=norm_layer,
                                           use_dropout=use_dropout, n_blocks=opt.n_resblocks_next, gpu_ids=gpu_ids,
                                           n_downsampling=opt.n_downsample, n_upsampling=opt.n_upsample, opt=opt)
            setattr(self, 'net' + str(self.num_G - 1 - i), netG)
            if self.opt.alpha_gate != '':
                if opt.alpha_gate == '0_1':
                    gate = AlphaGate(6, opt.not_mono_gate, opt.alpha_gate)
                elif opt.alpha_gate == '0_1_2':
                    gate = AlphaGate(6 + 3, opt.not_mono_gate, opt.alpha_gate)
                elif opt.alpha_gate == '1_2':
                    gate = AlphaGate(6, opt.not_mono_gate, opt.alpha_gate)
                elif opt.alpha_gate == 'simp':
                    gate = AlphaGateSimp()
                elif opt.alpha_gate == 'trans_2':
                    gate = AlphaGate(input_nc_new, opt.not_mono_gate, opt.alpha_gate)
                elif opt.alpha_gate == 'trans_1_2':
                    gate = AlphaGate(input_nc_new + 3, opt.not_mono_gate, opt.alpha_gate)
                setattr(self, 'gate' + str(self.num_G - 1 - i), gate)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, input):
        # import ipdb; ipdb.set_trace()
        if isinstance(input, (list, tuple)):
            input = input[0]
        G_input = []
        ret = []
        tmp = input
        for i in range(self.num_G):
            G_input.append(tmp)
            tmp = self.downsample(tmp)
        x = None
        for i in range(self.num_G - 1, -1, -1):
            # netG = getattr(self, 'net'+str(i))
            netG = 'self.net%d' % i
            if i == self.num_G - 1:
                x = eval('%s(G_input[i])' % netG)
                ret.append(x)
            else:
                x = self.upsample(x)
                if self.opt.half_add:
                    y = eval('%s(torch.cat((x, G_input[i]), 1)) * 0.5 + x * 0.5' % netG)
                elif self.opt.alpha_gate != '':
                    gate = 'self.gate%d' % i
                    y = eval(
                        '%s.forward(x_0 = self.upsample(ret[-1]), x_1 = %s(torch.cat((x, G_input[i]), 1)), t = G_input[i])' % (
                            gate, netG))
                else:
                    y = eval('%s(torch.cat((x, G_input[i]), 1))' % netG)
                ret.append(y)
                if self.opt.dense:
                    x = torch.cat((x, y), 1)
                else:
                    x = y
        ret.reverse()
        return ret


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', n_downsampling=2, n_upsampling=2, opt=None):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.alpha_feat = -1
        padding_type = 'reflect' if opt.not_caffe else 'zero'
        if opt.alpha_gate == '0_1':
            self.gate = AlphaGate(6, opt.not_mono_gate, opt.alpha_gate)
        elif opt.alpha_gate == '0_1_2':
            self.gate = AlphaGate(9, opt.not_mono_gate, opt.alpha_gate)
        elif opt.alpha_gate == '1_2':
            self.gate = AlphaGate(6, opt.not_mono_gate, opt.alpha_gate)
        elif opt.alpha_gate == 'simp':
            self.gate = AlphaGateSimp()

        if opt.use_lrelu:
            activation = nn.LeakyReLU(0.2, True)
        else:
            activation = nn.ReLU(True)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if opt.not_caffe:
            model = [nn.ReflectionPad2d(3),
                     nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                               bias=use_bias)]
        else:
            model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3,
                               bias=use_bias)]
        if not opt.use_dense:
            model += [norm_layer(ngf),
                      activation]

        for i in range(n_downsampling):
            mult = min(2 ** i, opt.max_ngf / ngf)
            if mult == opt.max_ngf / ngf:
                fac = 1
            else:
                fac = 2
            if opt.use_dense:
                model += [DenseBlock(ngf * mult, ngf * mult / 4, 12, BasicBlock),
                          TransitionBlock(ngf * mult * 4, ngf * mult * 2)]
            elif opt.downsample_7:
                model += [nn.Conv2d(ngf * mult, ngf * mult * fac, kernel_size=7,
                                    stride=2, padding=3, bias=use_bias),
                          norm_layer(ngf * mult * fac),
                          activation]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * fac, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * fac),
                          activation]

        mult = min(2 ** n_downsampling, opt.max_ngf / ngf)
        for i in range(n_blocks):
            if opt.use_dense:
                model += [DenseBlock(ngf * mult, ngf * mult / 4, 4, BasicBlock),
                          TransitionBlock(ngf * mult * 2, ngf * mult, down=False)]
            else:
                model += [
                    ResnetBlock(int(ngf * mult), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, shallow=opt.shallow_resblock, use_lrelu=opt.use_lrelu)]

        for i in range(n_upsampling):
            if opt.upsample == 'conv':
                model += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          activation]
                mult /= 2.0
            elif opt.upsample == 'conv_keep':
                model += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult)),
                          activation]
            elif opt.upsample == 'subpix':
                model += [nn.Conv2d(int(ngf * mult), int(ngf * mult * 2),
                                    kernel_size=3, stride=1, padding=1,
                                    bias=use_bias),
                          nn.PixelShuffle(2),
                          norm_layer(int(ngf * mult / 2)),
                          activation]
                mult /= 2.0
            elif opt.upsample == 'subpix_keep':
                model += [nn.Conv2d(int(ngf * mult), int(ngf * mult * 4),
                                    kernel_size=3, stride=1, padding=1,
                                    bias=use_bias),
                          nn.PixelShuffle(2),
                          norm_layer(int(ngf * mult)),
                          activation]
            elif opt.upsample == 'resize':
                model += [nn.UpsamplingBilinear2d(scale_factor=2),
                          nn.Conv2d(int(ngf * mult), int(ngf * mult / 2),
                                    kernel_size=3, stride=1, padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          activation]
                mult /= 2.0
            elif opt.upsample == 'resize_keep':
                model += [nn.UpsamplingBilinear2d(scale_factor=2),
                          nn.Conv2d(int(ngf * mult), int(ngf * mult),
                                    kernel_size=3, stride=1, padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult)),
                          activation]
            elif opt.upsample == 'subpix_4':
                if opt.use_dense:
                    model += [DenseBlock(int(ngf * mult), int(ngf * mult / 4.0), 12, BasicBlock),
                              TransitionBlock(ngf * mult * 4, ngf * mult * 2, down=False),
                              nn.PixelShuffle(2)]
                else:
                    model += [nn.Conv2d(int(ngf * mult) if i == 0 else int(ngf * mult * 2), int(ngf * mult * 4),
                                        kernel_size=3, stride=1, padding=1,
                                        bias=use_bias),
                              nn.PixelShuffle(2),
                              norm_layer(int(ngf * mult)),
                              activation]
                if i != n_upsampling - 1:
                    mult /= 2.0

        if opt.simple_block:
            last_nc = int(ngf * mult)
            for i in range(opt.simple_conv):
                model += [nn.Conv2d(last_nc, last_nc / 4, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(last_nc / 4),
                          activation]
                last_nc /= 4
            model += [nn.Conv2d(last_nc, output_nc, kernel_size=1, padding=0)]
            if opt.tanh_out:
                model += [nn.Tanh()]
        elif opt.not_caffe:
            model += [nn.ReflectionPad2d(3)]
            last_nc = int(ngf * mult)
            model += [nn.Conv2d(last_nc, output_nc, kernel_size=7, padding=0)]
            model += [nn.Tanh()]
        else:
            last_nc = int(ngf * mult)
            model += [nn.Conv2d(last_nc, output_nc, kernel_size=7, padding=3)]
            model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, y=None, t=None, cond=None):
        if y is not None:
            self.out = self.model(input)
            if t is not None:
                return self.gate.forward(self.out, y, t)
            else:
                return self.gate.forward(self.out, y)
        elif cond is not None:
            x = torch.cat([input, cond], dim=1)
            return self.model(x)
        else:
            return self.model(input)

    def forward_feat(self, input, feat, feat_only=False):
        if feat_only:
            x = feat
        else:
            x = input
            for idx in range(4):
                x = self.model[idx](x)
            x = torch.cat((x, feat), 1)

        for idx in range(4, len(self.model) - 3):
            x = self.model[idx](x)
        self.out_feat = x

        for idx in range(len(self.model) - 3, len(self.model)):
            x = self.model[idx](x)
        self.out = x
        return self.out, self.out_feat

    def get_feature(self, input, multi_out=False, feature_layer=3):
        y = input
        for idx in range(len(self.model) - feature_layer):
            y = self.model[idx](y)
        self.feat = y
        for idx in range(len(self.model) - feature_layer, len(self.model)):
            y = self.model[idx](y)
        self.out = y
        return self.out, self.feat

#Encoder of G
class ResnetEncoder_my(nn.Module):
    def __init__(self, input_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', n_downsampling=2,  opt=None):
        assert (n_blocks >= 0)
        super(ResnetEncoder_my, self).__init__() 
        self.input_nc = input_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.alpha_feat = -1
	self.n_downsampling = n_downsampling
        padding_type = 'reflect' if opt.not_caffe else 'zero'
        if opt.alpha_gate == '0_1':
            self.gate = AlphaGate(6, opt.not_mono_gate, opt.alpha_gate)
        elif opt.alpha_gate == '0_1_2':
            self.gate = AlphaGate(9, opt.not_mono_gate, opt.alpha_gate)
        elif opt.alpha_gate == '1_2':
            self.gate = AlphaGate(6, opt.not_mono_gate, opt.alpha_gate)
        elif opt.alpha_gate == 'simp':
            self.gate = AlphaGateSimp()

        if opt.use_lrelu:
            activation = nn.LeakyReLU(0.2, True)
        else:
            activation = nn.ReLU(True)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if opt.not_caffe:
            model = [nn.ReflectionPad2d(3),
                     nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                               bias=use_bias)]
        else:
            model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3,
                               bias=use_bias)]
        if not opt.use_dense:
            model += [norm_layer(ngf),
                      activation]

        for i in range(n_downsampling):
            mult = min(2 ** i, opt.max_ngf / ngf)
            if mult == opt.max_ngf / ngf:
                fac = 1
            else:
                fac = 2
            if opt.use_dense:
                model += [DenseBlock(ngf * mult, ngf * mult / 4, 12, BasicBlock),
                          TransitionBlock(ngf * mult * 4, ngf * mult * 2)]
            elif opt.downsample_7:
                model += [nn.Conv2d(ngf * mult, ngf * mult * fac, kernel_size=7,
                                    stride=2, padding=3, bias=use_bias),
                          norm_layer(ngf * mult * fac),
                          activation]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * fac, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * fac),
                          activation]

        mult = min(2 ** n_downsampling, opt.max_ngf / ngf)
	self.mult = mult
        for i in range(n_blocks):
            if opt.use_dense:
                model += [DenseBlock(ngf * mult, ngf * mult / 4, 4, BasicBlock),
                          TransitionBlock(ngf * mult * 2, ngf * mult, down=False)]
            else:
                model += [
                    ResnetBlock(int(ngf * mult), padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, shallow=opt.shallow_resblock, use_lrelu=opt.use_lrelu)]

        self.model = nn.Sequential(*model)

    def get_mult(self):
	return self.mult

    def forward(self, input, y=None, t=None, cond=None):
        if y is not None:
            self.out = self.model(input)
            if t is not None:
                return self.gate.forward(self.out, y, t)
            else:
                return self.gate.forward(self.out, y)
        elif cond is not None:
            x = torch.cat([input, cond], dim=1)
            return self.model(x)
        else:
            return self.model(input)

    def forward_feat(self, input, feat, feat_only=False):
        if feat_only:
            x = feat
        else:
            x = input
            for idx in range(4):
                x = self.model[idx](x)
            x = torch.cat((x, feat), 1)

        for idx in range(4, len(self.model) - 3):
            x = self.model[idx](x)
        self.out_feat = x

        for idx in range(len(self.model) - 3, len(self.model)):
            x = self.model[idx](x)
        self.out = x
        return self.out, self.out_feat

    def get_feature(self, input, multi_out=False, feature_layer=3):
        y = input
        for idx in range(len(self.model) - feature_layer):
            y = self.model[idx](y)
        self.feat = y
        for idx in range(len(self.model) - feature_layer, len(self.model)):
            y = self.model[idx](y)
        self.out = y
        return self.out, self.feat


#Decoder infront
class ResnetDecoder_my(nn.Module):
    def __init__(self, mult,  ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', n_upsampling=2, opt=None):
        assert (n_blocks >= 0)
        super(ResnetDecoder_my, self).__init__()
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.alpha_feat = -1
        padding_type = 'reflect' if opt.not_caffe else 'zero'
        if opt.alpha_gate == '0_1':
            self.gate = AlphaGate(6, opt.not_mono_gate, opt.alpha_gate)
        elif opt.alpha_gate == '0_1_2':
            self.gate = AlphaGate(9, opt.not_mono_gate, opt.alpha_gate)
        elif opt.alpha_gate == '1_2':
            self.gate = AlphaGate(6, opt.not_mono_gate, opt.alpha_gate)
        elif opt.alpha_gate == 'simp':
            self.gate = AlphaGateSimp()

        if opt.use_lrelu:
            activation = nn.LeakyReLU(0.2, True)
        else:
            activation = nn.ReLU(True)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        for i in range(n_upsampling):
            if opt.upsample == 'conv':
                model += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          activation]
                mult /= 2.0
            elif opt.upsample == 'conv_keep':
                model += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult)),
                          activation]
            elif opt.upsample == 'subpix':
                model += [nn.Conv2d(int(ngf * mult), int(ngf * mult * 2),
                                    kernel_size=3, stride=1, padding=1,
                                    bias=use_bias),
                          nn.PixelShuffle(2),
                          norm_layer(int(ngf * mult / 2)),
                          activation]
                mult /= 2.0
            elif opt.upsample == 'subpix_keep':
                model += [nn.Conv2d(int(ngf * mult), int(ngf * mult * 4),
                                    kernel_size=3, stride=1, padding=1,
                                    bias=use_bias),
                          nn.PixelShuffle(2),
                          norm_layer(int(ngf * mult)),
                          activation]
            elif opt.upsample == 'resize':
                model += [nn.UpsamplingBilinear2d(scale_factor=2),
                          nn.Conv2d(int(ngf * mult), int(ngf * mult / 2),
                                    kernel_size=3, stride=1, padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          activation]
                mult /= 2.0
            elif opt.upsample == 'resize_keep':
                model += [nn.UpsamplingBilinear2d(scale_factor=2),
                          nn.Conv2d(int(ngf * mult), int(ngf * mult),
                                    kernel_size=3, stride=1, padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult)),
                          activation]
            elif opt.upsample == 'subpix_4':
                if opt.use_dense:
                    model += [DenseBlock(int(ngf * mult), int(ngf * mult / 4.0), 12, BasicBlock),
                              TransitionBlock(ngf * mult * 4, ngf * mult * 2, down=False),
                              nn.PixelShuffle(2)]
                else:
                    model += [nn.Conv2d(int(ngf * mult) if i == 0 else int(ngf * mult * 2), int(ngf * mult * 4),
                                        kernel_size=3, stride=1, padding=1,
                                        bias=use_bias),
                              nn.PixelShuffle(2),
                              norm_layer(int(ngf * mult)),
                              activation]
                if i != n_upsampling - 1:
                    mult /= 2.0
	self.model = nn.Sequential(*model)
	self.mult = mult

    def get_mult(self):
	return self.mult

    def forward(self, input, y=None, t=None, cond=None):
        if y is not None:
            self.out = self.model(input)
            if t is not None:
                return self.gate.forward(self.out, y, t)
            else:
                return self.gate.forward(self.out, y)
        elif cond is not None:
            x = torch.cat([input, cond], dim=1)
            return self.model(x)
        else:
            return self.model(input)

    def forward_feat(self, input, feat, feat_only=False):
        if feat_only:
            x = feat
        else:
            x = input
            for idx in range(4):
                x = self.model[idx](x)
            x = torch.cat((x, feat), 1)

        for idx in range(4, len(self.model) - 3):
            x = self.model[idx](x)
        self.out_feat = x

        for idx in range(len(self.model) - 3, len(self.model)):
            x = self.model[idx](x)
        self.out = x
        return self.out, self.out_feat

    def get_feature(self, input, multi_out=False, feature_layer=3):
        y = input
        for idx in range(len(self.model) - feature_layer):
            y = self.model[idx](y)
        self.feat = y
        for idx in range(len(self.model) - feature_layer, len(self.model)):
            y = self.model[idx](y)
        self.out = y
        return self.out, self.feat


#Last layer of Generator
class GeneratorLL(nn.Module):
    def __init__(self, mult, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', opt=None):
        assert (n_blocks >= 0)
        super(GeneratorLL, self).__init__()
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.alpha_feat = -1
        padding_type = 'reflect' if opt.not_caffe else 'zero'
        if opt.alpha_gate == '0_1':
            self.gate = AlphaGate(6, opt.not_mono_gate, opt.alpha_gate)
        elif opt.alpha_gate == '0_1_2':
            self.gate = AlphaGate(9, opt.not_mono_gate, opt.alpha_gate)
        elif opt.alpha_gate == '1_2':
            self.gate = AlphaGate(6, opt.not_mono_gate, opt.alpha_gate)
        elif opt.alpha_gate == 'simp':
            self.gate = AlphaGateSimp()

        if opt.use_lrelu:
            activation = nn.LeakyReLU(0.2, True)
        else:
            activation = nn.ReLU(True)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        if opt.simple_block:
            last_nc = int(ngf * mult)
            for i in range(opt.simple_conv):
                model += [nn.Conv2d(last_nc, last_nc / 4, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(last_nc / 4),
                          activation]
                last_nc /= 4
            model += [nn.Conv2d(last_nc, output_nc, kernel_size=1, padding=0)]
            if opt.tanh_out:
                model += [nn.Tanh()]
        elif opt.not_caffe:
            model += [nn.ReflectionPad2d(3)]
            last_nc = int(ngf * mult)
            model += [nn.Conv2d(last_nc, output_nc, kernel_size=7, padding=0)]
            model += [nn.Tanh()]
        else:
            last_nc = int(ngf * mult)
            model += [nn.Conv2d(last_nc, output_nc, kernel_size=7, padding=3)]
            model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, y=None, t=None, cond=None):
        if y is not None:
            self.out = self.model(input)
            if t is not None:
                return self.gate.forward(self.out, y, t)
            else:
                return self.gate.forward(self.out, y)
        elif cond is not None:
            x = torch.cat([input, cond], dim=1)
            return self.model(x)
        else:
            return self.model(input)

    def forward_feat(self, input, feat, feat_only=False):
        if feat_only:
            x = feat
        else:
            x = input
            for idx in range(4):
                x = self.model[idx](x)
            x = torch.cat((x, feat), 1)

        for idx in range(4, len(self.model) - 3):
            x = self.model[idx](x)
        self.out_feat = x

        for idx in range(len(self.model) - 3, len(self.model)):
            x = self.model[idx](x)
        self.out = x
        return self.out, self.out_feat

    def get_feature(self, input, multi_out=False, feature_layer=3):
        y = input
        for idx in range(len(self.model) - feature_layer):
            y = self.model[idx](y)
        self.feat = y
        for idx in range(len(self.model) - feature_layer, len(self.model)):
            y = self.model[idx](y)
        self.out = y
        return self.out, self.feat


class Generator_end2end(nn.Module):
    def __init__(self, opt=None):
        super(Generator_end2end, self).__init__()
        input_nc = opt.input_nc
        output_nc = opt.output_nc
        ngf = opt.ngf
        n_downsampling = opt.n_downsample
        n_upsampling = opt.n_upsample
        use_dropout = not opt.no_dropout
        n_blocks = opt.n_resblocks
        padding_type = 'reflect'
        self.out_num = opt.out_num

        if opt.use_lrelu:
            activation = nn.LeakyReLU(0.2, True)
        else:
            activation = nn.ReLU(True)

        norm_layer = nn.InstanceNorm2d
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 activation]

        for i in range(n_downsampling):
            mult = min(2 ** i, opt.max_ngf / ngf)
            if mult == opt.max_ngf / ngf:
                fac = 1
            else:
                fac = 2
            model += [nn.Conv2d(ngf * mult, ngf * mult * fac, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * fac),
                      activation,
                      ResnetBlock(ngf * mult * fac, padding_type=padding_type, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, use_lrelu=opt.use_lrelu)]

        mult = min(2 ** n_downsampling, opt.max_ngf / ngf)
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias, use_lrelu=opt.use_lrelu)]

        self.up_proc = []
        self.out_trans = []
        for i in range(n_upsampling):
            self.up_proc.append(nn.Sequential(*[nn.Conv2d(int(ngf * mult), int(ngf * mult * 2),
                                                          kernel_size=3, stride=1, padding=1,
                                                          bias=use_bias),
                                                nn.PixelShuffle(2),
                                                norm_layer(int(ngf * mult / 2)),
                                                activation,
                                                ResnetBlock(int(ngf * mult / 2), padding_type=padding_type,
                                                            norm_layer=norm_layer, use_dropout=use_dropout,
                                                            use_bias=use_bias, use_lrelu=opt.use_lrelu)]))
            mult /= 2.0
            if i - (n_upsampling - self.out_num - 1) > 0:
                self.out_trans.append(nn.Sequential(*[nn.ReflectionPad2d(3),
                                                      nn.Conv2d(int(ngf * mult), output_nc, kernel_size=7, padding=0),
                                                      nn.Tanh()]))

        self.model = nn.Sequential(*model)
        for i in self.up_proc:
            i = i.cuda()
        for i in self.out_trans:
            i = i.cuda()

    def forward(self, input):
        y = self.model(input)
        for i in range(len(self.up_proc)):
            y = self.up_proc[i](y)
        return self.out_trans[-1](y)

    def forward_get_idx(self, input, idx):
        y = self.model(input)
        assert (idx < self.out_num)
        for i in range(len(self.up_proc) - idx):
            y = self.up_proc[i](y)
        return self.out_trans[-(1 + idx)](y)

    def forward_multi(self, input):
        y = self.model(input)
        ret = []
        for i in range(len(self.up_proc)):
            y = self.up_proc[i](y)
            if i - (len(self.up_proc) - self.out_num) >= 0:
                ret.append(self.out_trans[i - (len(self.up_proc) - self.out_num)](y))
        return ret


class Generator_stack_simp(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', n_downsampling=2, n_upsampling=2, side='A', opt=None):
        assert (n_blocks >= 0)
        super(Generator_stack_simp, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.alpha_feat = -1
        self.opt = opt

        if opt.alpha_gate == '0_1':
            self.gate = AlphaGate(6, opt.not_mono_gate, opt.alpha_gate)
        elif opt.alpha_gate == '0_1_2':
            self.gate = AlphaGate(9, opt.not_mono_gate, opt.alpha_gate)
        elif opt.alpha_gate == '1_2':
            self.gate = AlphaGate(6, opt.not_mono_gate, opt.alpha_gate)
        elif 'simp' in opt.alpha_gate:
            self.gate = AlphaGateSimp(gate_type=opt.alpha_gate)

        if opt.use_lrelu:
            activation = nn.LeakyReLU(0.2, True)
        else:
            activation = nn.ReLU(True)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if opt.stack_keep_nc:
            next_nc = ngf
        else:
            next_nc = int(ngf // 2)

        mid_nc = next_nc

        model_in = [nn.ConvTranspose2d(int(ngf), int(next_nc),
                                       kernel_size=3, stride=2,
                                       padding=1, output_padding=1,
                                       bias=use_bias),
                    norm_layer(int(next_nc)),
                    activation]

        if opt.stack_simp_conv:
            model_mid = [nn.ReflectionPad2d(1),
                         nn.Conv2d(mid_nc + (self.input_nc if opt.stack_imgin else 0), mid_nc, kernel_size=3, padding=0,
                                   bias=use_bias),
                         norm_layer(mid_nc),
                         activation, ]
            if use_dropout:
                model_mid += [nn.Dropout(0.5), ]
            model_mid += [nn.ReflectionPad2d(1),
                          nn.Conv2d(mid_nc, mid_nc, kernel_size=3, padding=0, bias=use_bias),
                          norm_layer(mid_nc),
                          activation, ]
        else:
            model_mid = [ResnetBlock(int(mid_nc) + (self.input_nc if opt.stack_imgin else 0), padding_type=padding_type,
                                     norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias,
                                     shallow=opt.shallow_resblock, use_lrelu=opt.use_lrelu)]
            for i in range(n_blocks - 1):
                model_mid += [
                    ResnetBlock(int(mid_nc) + (self.input_nc if opt.stack_imgin else 0), padding_type=padding_type,
                                norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias,
                                shallow=opt.shallow_resblock, use_lrelu=opt.use_lrelu)]
            mid_nc += (self.input_nc if opt.stack_imgin else 0)

        if not opt.simple_block:
            model_out = [nn.ReflectionPad2d(3)]
            model_out += [nn.Conv2d(mid_nc, output_nc, kernel_size=7, padding=0)]
            model_out += [nn.Tanh()]
        else:
            model_out = [nn.ReflectionPad2d(0)]
            model_out += [nn.Conv2d(mid_nc, output_nc, kernel_size=1, padding=0)]
            model_out += [nn.Tanh()]

        self.model_in = nn.Sequential(*model_in)
        self.model_mid = nn.Sequential(*model_mid)
        self.model_out = nn.Sequential(*model_out)
        self.sub_model = define_G(opt.pre.input_nc, opt.pre.output_nc, opt.pre.ngf, opt.pre.which_model_netG,
                                  opt.pre.norm, not opt.pre.no_dropout, opt.pre.init_type, self.gpu_ids,
                                  n_downsampling=opt.pre.n_downsample, n_resblocks=opt.pre.n_resblocks, side=side,
                                  opt=opt.pre)
        print_network(self.sub_model, opt, input_shape=(opt.pre.input_nc, opt.pre.fineSize, opt.pre.fineSize))
        if not opt.idt:
            self.down_2 = torch.nn.AvgPool2d(2)
            self.up_2 = torch.nn.Upsample(scale_factor=2)
        else:
            self.down_2 = torch.nn.AvgPool2d(1)
            self.up_2 = torch.nn.AvgPool2d(1)

    def forward_feat(self, input, img=None):
        x = input
        x = self.model_in(x)
        if img is not None:
            x = torch.cat((x, img), 1)
        x = self.model_mid(x)
        self.out_feat = x
        x = self.model_out(x)
        self.out = x
        return self.out, self.out_feat

    def forward(self, input, multi_out=False):
        if isinstance(input, (list, tuple)):
            t = input[0]
        else:
            t = input
        self.x_0, feat = self.sub_model.get_feature(self.down_2(t), multi_out, self.opt.sub_out)
        if self.opt.stack_imgin:
            self.out, self.feat = self.forward_feat(feat, t)
        else:
            self.out, self.feat = self.forward_feat(feat)
        if self.opt.alpha_gate != '':
            ret = self.gate.forward(x_0=self.up_2(self.x_0), x_1=self.out, t=t)
        else:
            ret = self.out
        if multi_out:
            return ret, self.x_0
        else:
            return ret

    def get_feature(self, input, multi_out=False):
        ret = self.forward(input, multi_out)
        return ret, self.feat


class Generator_stack(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', n_downsampling=2, n_upsampling=2, side='A', opt=None):
        assert (n_blocks >= 0)
        super(Generator_stack, self).__init__()
        if opt.skip_feat or opt.img_only:
            self.input_nc = input_nc
        else:
            self.input_nc = input_nc + opt.pre.output_nc
            input_nc = self.input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.alpha_feat = -1
        self.opt = opt

        if opt.alpha_gate == '0_1':
            self.gate = AlphaGate(6, opt.not_mono_gate, opt.alpha_gate)
        elif opt.alpha_gate == '0_1_2':
            self.gate = AlphaGate(9, opt.not_mono_gate, opt.alpha_gate)
        elif opt.alpha_gate == '1_2':
            self.gate = AlphaGate(6, opt.not_mono_gate, opt.alpha_gate)
        elif 'simp' in opt.alpha_gate:
            self.gate = AlphaGateSimp(gate_type=opt.alpha_gate)

        if opt.use_lrelu:
            activation = nn.LeakyReLU(0.2, True)
        else:
            activation = nn.ReLU(True)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if not opt.simple_block:
            model = [nn.ReflectionPad2d(3),
                     nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                               bias=use_bias),
                     norm_layer(ngf),
                     activation]
            if opt.skip_feat and not self.opt.feat_only:
                model += [nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * 2),
                          activation]
            else:  # skip_img ot feat_only
                model += [nn.Conv2d(ngf, ngf * 2, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * 2),
                          activation]
        else:
            assert (n_downsampling == 0 and n_upsampling == 0)
            model = [nn.ReflectionPad2d(0),
                     nn.Conv2d(input_nc, ngf, kernel_size=1, padding=0,
                               bias=use_bias),
                     norm_layer(ngf),
                     activation]
            if opt.skip_feat and not self.opt.feat_only:
                model += [nn.Conv2d(ngf * 2, ngf, kernel_size=3,
                                    stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf),
                          activation,
                          nn.Conv2d(ngf, int(ngf // 2), kernel_size=3,
                                    stride=1, padding=1, bias=use_bias),
                          norm_layer(int(ngf // 2)),
                          activation, ]
            else:  # skip_img ot feat_only
                model += [nn.Conv2d(ngf, int(ngf // 2), kernel_size=3,
                                    stride=1, padding=1, bias=use_bias),
                          norm_layer(int(ngf // 2)),
                          activation]

        for i in range(1, n_downsampling):
            mult = min(2 ** i, opt.max_ngf / ngf)
            if mult == opt.max_ngf / ngf:
                fac = 1
            else:
                fac = 2
            model += [nn.Conv2d(ngf * mult, ngf * mult * fac, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * fac),
                      activation]

        if not opt.simple_block:
            mult = min(2 ** n_downsampling, opt.max_ngf / ngf)
            for i in range(n_blocks):
                model += [
                    ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                use_bias=use_bias, shallow=opt.shallow_resblock, use_lrelu=opt.use_lrelu)]
        else:
            for i in range(n_blocks):
                model += [ResnetBlock(int(ngf // 2), padding_type=padding_type, norm_layer=norm_layer,
                                      use_dropout=use_dropout, use_bias=use_bias, shallow=opt.shallow_resblock,
                                      use_lrelu=opt.use_lrelu)]
            last_nc = int(ngf // 2)

        for i in range(n_upsampling):
            model += [nn.Conv2d(int(ngf * mult), int(ngf * mult * 2),
                                kernel_size=3, stride=1, padding=1,
                                bias=use_bias),
                      nn.PixelShuffle(2),
                      norm_layer(int(ngf * mult / 2)),
                      activation]
            mult /= 2.0
            last_nc = int(ngf * mult)

        if not opt.simple_block or opt.out7:
            model += [nn.ReflectionPad2d(3)]
            model += [nn.Conv2d(last_nc, output_nc, kernel_size=7, padding=0)]
            model += [nn.Tanh()]
        else:
            model += [nn.ReflectionPad2d(0)]
            model += [nn.Conv2d(last_nc, output_nc, kernel_size=1, padding=0)]
            model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.sub_model = define_G(opt.pre.input_nc, opt.pre.output_nc, opt.pre.ngf, opt.pre.which_model_netG,
                                  opt.pre.norm, not opt.pre.no_dropout, opt.pre.init_type, self.gpu_ids,
                                  n_downsampling=opt.pre.n_downsample, n_resblocks=opt.pre.n_resblocks, side=side,
                                  opt=opt.pre)
        print_network(self.sub_model, opt, input_shape=(opt.pre.input_nc, opt.pre.fineSize, opt.pre.fineSize))
        if not opt.idt:
            self.down_2 = torch.nn.AvgPool2d(2)
            self.up_2 = torch.nn.Upsample(scale_factor=2)
        else:
            self.down_2 = torch.nn.AvgPool2d(1)
            self.up_2 = torch.nn.AvgPool2d(1)

    def forward_feat(self, input, feat=None, feat_only=False):
        if feat_only:
            x = feat
        else:
            x = input
            for idx in range(4):
                x = self.model[idx](x)
            if feat is not None:
                x = torch.cat((x, feat), 1)

        for idx in range(4, len(self.model) - 3):
            x = self.model[idx](x)
        self.out_feat = x

        for idx in range(len(self.model) - 3, len(self.model)):
            x = self.model[idx](x)
        self.out = x

        return self.out, self.out_feat

    def get_out(self, t):
        self.x_0, feat = self.sub_model.get_feature(self.down_2(t))
        if self.opt.skip_feat:
            self.out, self.feat = self.forward_feat(t, self.up_2(feat), self.opt.feat_only)
        elif not self.opt.img_only:
            self.out, self.feat = self.forward_feat(torch.cat((self.up_2(self.x_0), t), 1))
        else:
            self.out, self.feat = self.forward_feat(self.up_2(self.x_0))
        if self.opt.alpha_gate != '':
            return self.gate.forward(x_0=self.up_2(self.x_0), x_1=self.out, t=t)
        else:
            return self.out

    def forward(self, input, multi_out=False):
        # import ipdb; ipdb.set_trace()
        ret = self.get_out(input)
        if multi_out:
            return ret, self.x_0
        else:
            return ret

    def get_feature(self, input):
        ret = self.get_out(input)
        return ret, self.feat


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[],
                 one_out=None, feat_len=None, kw=None, opt=None):
        super(NLayerDiscriminator, self).__init__()
        one_out = opt.one_out if one_out is None else one_out
        feat_len = opt.feat_len_D if feat_len is None else feat_len
        kw = opt.kw if kw is None else kw
        self.gpu_ids = gpu_ids
        self.alpha_feat = -1
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult)]
            if opt.use_noise:
                sequence += [GaussianNoise()]
            sequence += [nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        if not one_out and feat_len != 0 or opt.legacy_D:
            nf_mult = min(2 ** n_layers, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

            if one_out:
                sequence += [Flatten(), nn.Linear(feat_len, 1)]
        else:
            for n in range(1, n_layers):
                nf_mult = min(2 ** (n_layers - n - 1), 8)
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=3, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
                nf_mult_prev = nf_mult

            if one_out:
                sequence += [Flatten(), nn.Linear(ndf * nf_mult * feat_len, 1)]
            else:
                sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

    def forward_feat(self, input, feat):
        x = input
        for idx in range(2):
            x = self.model[idx](x)

        if self.alpha_feat > 0:
            x = (1 - self.alpha_feat) * x + self.alpha_feat * feat
        else:
            x = feat

        for idx in range(2, len(self.model)):
            x = self.model[idx](x)
        self.out = x
        return self.out

    def get_feat(self, input, feat):
        x = input
        for idx in range(2):
            x = self.model[idx](x)

        if self.alpha_feat > 0:
            x = (1 - self.alpha_feat) * x + self.alpha_feat * feat
        else:
            x = feat

        for idx in range(2, len(self.model)):
            x = self.model[idx](x)
        self.out = x
        return self.out

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerAttrDiscriminator(nn.Module):
    def __init__(self, input_nc, n_cond=1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[],
                 one_out=None, feat_len=None, kw=None, opt=None):
        super(NLayerAttrDiscriminator, self).__init__()
        feat_len = opt.feat_len_D if feat_len is None else feat_len
        kw = opt.kw if kw is None else kw
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 3
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult)]
            if opt.use_noise:
                sequence += [GaussianNoise()]
            sequence += [nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        self.model = nn.Sequential(*sequence)
        self.conv1 = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)

        sequence2 = []
        # for n in range(n_layers, n_layers + 2):
        #     nf_mult_prev = nf_mult
        #     nf_mult = min(2 ** n, 8)
        #     sequence2 += [
        #         nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
        #                   kernel_size=kw, stride=2, padding=padw, bias=use_bias),
        #         norm_layer(ndf * nf_mult),
        #         nn.LeakyReLU(0.2, True)
        #     ]
        sequence2 += [nn.Conv2d(ndf * nf_mult, n_cond, kernel_size=kw, stride=2, padding=padw)]
        self.conv2 = nn.Sequential(*sequence2)

    def forward(self, input):
        h = self.model(input)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls#.view(out_cls.size(0), out_cls.size(1))


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_D=3, opt=None):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, opt=opt)
            setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


class MultiDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_D=3, opt=None):
        super(MultiDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.single_D = opt.single_D
        self.pix2pix_D = opt.pix2pix_D

        if self.single_D:
            self.netD = NLayerDiscriminator(input_nc * self.num_D, ndf, n_layers, norm_layer, use_sigmoid, opt=opt)
        else:
            for i in range(num_D):
                if not self.pix2pix_D or (self.pix2pix_D and i == 0):
                    netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, opt=opt)
                else:
                    netD = NLayerDiscriminator(input_nc * 2, ndf, n_layers, norm_layer, use_sigmoid, opt=opt)
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')

    def singleD_forward(self, model, input):
        return [eval('%s(input)' % model)]

    def forward(self, input):
        num_D = self.num_D
        if not isinstance(input, (list, tuple)):
            tmp = [input, ]
            for i in range(num_D - 1):
                tmp.append(self.downsample(tmp[-1]))
            input_new = tuple(tmp)
        else:
            input_new = input
        result = []
        if self.single_D:
            input_new = list(input_new)
            for i in range(self.num_D):
                for j in range(i):
                    input_new[i] = self.upsample(input_new[i])
            input_new = tuple(input_new)
            input_new = torch.cat(input_new, 1)
            result.append(self.singleD_forward('self.netD', input_new))
        else:
            for i in range(num_D):
                model = 'self.layer%d' % (num_D - 1 - i)
                if not self.pix2pix_D or (self.pix2pix_D and i == (num_D - 1)):
                    result.append(self.singleD_forward(model, input_new[i]))
                else:
                    result.append(
                        self.singleD_forward(model, torch.cat((input_new[i], self.upsample(input_new[i + 1])), 1)))
        return result


class Generator_core(nn.Module):
    def __init__(self, input_nc, output_nc, mid_nc=64, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', n_downsampling=1, n_upsampling=1, opt=None):
        assert (n_blocks >= 0)
        super(Generator_core, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.alpha_feat = 0.0
        self.mid_net = None
        if opt.use_lrelu:
            activation = nn.LeakyReLU(0.2, True)
        else:
            activation = nn.ReLU(True)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        in_trans = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)]

        pre_model = [norm_layer(ngf),
                     activation]
        for i in range(n_downsampling):
            mult = min(2 ** i, opt.max_ngf / ngf)
            if mult == opt.max_ngf / ngf:
                fac = 1
            else:
                fac = 2
            pre_model += [nn.Conv2d(ngf * mult, ngf * mult * fac, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * fac),
                          activation]

        mult = min(2 ** n_downsampling, opt.max_ngf / ngf)
        for i in range(n_blocks // 2):
            pre_model += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias, shallow=opt.shallow_resblock, use_lrelu=opt.use_lrelu)]
        pre_model += [nn.Conv2d(ngf * mult, mid_nc, kernel_size=1, padding=0),
                      norm_layer(mid_nc),
                      activation]

        post_model = [nn.Conv2d(mid_nc, ngf * mult, kernel_size=1, padding=0),
                      norm_layer(ngf * mult),
                      activation]
        for i in range(n_blocks // 2):
            post_model += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias, shallow=opt.shallow_resblock, use_lrelu=opt.use_lrelu)]

        for i in range(n_upsampling):
            post_model += [nn.Conv2d(int(ngf * mult), int(ngf * mult * 2),
                                     kernel_size=3, stride=1, padding=1,
                                     bias=use_bias),
                           nn.PixelShuffle(2),
                           norm_layer(int(ngf * mult / 2)),
                           activation]
            mult /= 2.0

        out_trans = [nn.ReflectionPad2d(3)]
        last_nc = int(ngf * mult)
        out_trans += [nn.Conv2d(last_nc, output_nc, kernel_size=7, padding=0)]
        out_trans += [nn.Tanh()]

        self.in_trans = nn.Sequential(*in_trans)
        self.out_trans = nn.Sequential(*out_trans)
        self.pre_model = nn.Sequential(*pre_model)
        self.post_model = nn.Sequential(*post_model)
        self.down_2 = nn.AvgPool2d(2 ** (n_downsampling))

    def forward_feat(self, input, feat=None):
        x = input
        x_down, x_pre = self.down_2(x), self.in_trans(x)
        if feat is not None:
            if self.alpha_feat > 0:
                pre_feat = self.pre_model((1 - self.alpha_feat) * x_pre + self.alpha_feat * feat)
            else:
                pre_feat = self.pre_model(feat)
        else:
            pre_feat = self.pre_model(x_pre)

        mid_out, mid_feat = self.mid_net.forward_feat(x_down, pre_feat)

        self.out_feat = self.post_model(mid_feat)
        self.out = self.out_trans(self.out_feat)

        if feat is not None:
            return self.out, self.out_feat
        else:
            return self.out, mid_out

    def forward(self, input):
        x = input
        x_down, x_pre = self.down_2(x), self.in_trans(x)

        pre_feat = self.pre_model(x_pre)

        mid_out, mid_feat = self.mid_net.forward_feat(x_down, pre_feat)

        self.out_feat = self.post_model(mid_feat)
        self.out = self.out_trans(self.out_feat)

        return self.out


class Discriminator_core(nn.Module):
    def __init__(self, out_nc, ndf=64, n_layers=1, norm_layer=nn.BatchNorm2d, opt=None):
        super(Discriminator_core, self).__init__()
        use_bias = True
        norm_layer = nn.InstanceNorm2d
        activation = nn.LeakyReLU(0.2, True)
        input_nc = 3
        kw = 3
        padw = 1
        self.next_net = None

        in_trans = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        model = []
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(0, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult)]
            if opt.use_noise:
                model += [GaussianNoise()]
            model += [nn.LeakyReLU(0.2, True)]

        model += [nn.Conv2d(ndf * nf_mult, out_nc, kernel_size=1, padding=0),
                  norm_layer(out_nc),
                  activation]

        self.in_trans = nn.Sequential(*in_trans)
        self.model = nn.Sequential(*model)
        self.down_2 = nn.AvgPool2d(2)

    def forward(self, input):
        x = input
        x_down, x_pre = self.down_2(x), self.in_trans(x)
        self.out = self.next_net.forward_feat(x_down, self.model(x_pre))
        return self.out

    def forward_feat(self, input, feat=None):
        x = input
        x_down, x_pre = self.down_2(x), self.in_trans(x)
        if feat is not None:
            if self.alpha_feat >= 0:
                pre_feat = self.model((1 - self.alpha_feat) * x_pre + self.alpha_feat * feat)
            else:
                pre_feat = self.model(feat)
        else:
            pre_feat = self.model(x_pre)

        self.out = self.next_net.forward_feat(x_down, pre_feat)
        return self.out


class Generator_stage2only(torch.nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], padding_type='reflect', opt=None):
        super(Generator_stage2only, self).__init__()
        self.gpu_ids = gpu_ids
        activation = nn.LeakyReLU(0.2, True)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = nn.InstanceNorm2d

        input_nc = 3
        output_nc = 3
        ngf = 64
        self.conv2_1 = NNBlock(input_nc, ngf, nn_type='down_conv7', norm=norm_layer, activation=activation,
                               dropout=False)
        self.conv2_2 = NNBlock(ngf, ngf * 2, nn_type='down_conv', norm=norm_layer, activation=activation, dropout=False)
        self.conv2_3 = NNBlock(ngf * 2, ngf * 4, nn_type='down_conv', norm=norm_layer, activation=activation,
                               dropout=False)

        ch = (ngf * 8)
        self.res5 = ResBlock(ch, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias,
                             shallow=opt.shallow_resblock)
        self.res6 = ResBlock(ch, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias,
                             shallow=opt.shallow_resblock)
        self.res7 = ResBlock(ch, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias,
                             shallow=opt.shallow_resblock)
        self.res8 = ResBlock(ch, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias,
                             shallow=opt.shallow_resblock)

        self.up2_1 = NNBlock(ch, ch // 2, nn_type='up_' + opt.upsample, norm=norm_layer, activation=activation,
                             dropout=False)
        self.up2_2 = NNBlock(ch // 2, ch // 4, nn_type='up_' + opt.upsample, norm=norm_layer, activation=activation,
                             dropout=False)
        self.out_conv2 = NNBlock(ch // 4, output_nc, nn_type='down_conv7', norm=None, activation=nn.Tanh())

    def forward(self, x, feat):
        z = self.conv2_1(x)
        z = self.conv2_2(z)
        z = self.conv2_3(z)

        y = torch.cat((feat, z), 1)

        y = self.res5(y)
        y = self.res6(y)
        y = self.res7(y)
        y = self.res8(y)

        y = self.up2_1(y)
        y = self.up2_2(y)
        self.out = self.out_conv2(y)
        return self.out


class Generator(torch.nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', n_downsampling=2, n_upsampling=2, opt=None):
        super(Generator, self).__init__()
        self.gpu_ids = gpu_ids
        ch = ngf
        activation = nn.ReLU(True)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = nn.InstanceNorm2d

        self.conv1_1 = NNBlock(input_nc, ngf, nn_type='down_conv7', norm=norm_layer, activation=activation,
                               dropout=False)
        self.conv1_2 = NNBlock(ngf, ngf * 2, nn_type='down_conv', norm=norm_layer, activation=activation, dropout=False)

        ch = ngf * 2
        self.res1 = ResBlock(ch, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias,
                             shallow=opt.shallow_resblock)
        self.res2 = ResBlock(ch, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias,
                             shallow=opt.shallow_resblock)
        self.res3 = ResBlock(ch, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias,
                             shallow=opt.shallow_resblock)
        self.res4 = ResBlock(ch, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias,
                             shallow=opt.shallow_resblock)
        self.res5 = ResBlock(ch, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias,
                             shallow=opt.shallow_resblock)

        self.up1_1 = NNBlock(ch, ch // 2, nn_type='up_' + opt.upsample, norm=norm_layer, activation=activation,
                             dropout=False)
        self.out_conv1 = NNBlock(ch // 2, output_nc, nn_type='down_conv7', norm=None, activation=nn.Tanh())

        self.conv2_1 = NNBlock(input_nc, ngf, nn_type='down_conv7', norm=norm_layer, activation=activation,
                               dropout=False)
        self.conv2_2 = NNBlock(ngf, ngf * 2, nn_type='down_conv', norm=norm_layer, activation=activation, dropout=False)
        self.conv2_3 = NNBlock(ngf * 2, ngf * 4, nn_type='down_conv', norm=norm_layer, activation=activation,
                               dropout=False)

        ch = ngf * 6
        self.res6 = ResBlock(ch, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias,
                             shallow=opt.shallow_resblock)
        self.res7 = ResBlock(ch, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias,
                             shallow=opt.shallow_resblock)
        self.res8 = ResBlock(ch, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias,
                             shallow=opt.shallow_resblock)
        self.res9 = ResBlock(ch, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias,
                             shallow=opt.shallow_resblock)

        self.up2_1 = NNBlock(ch, ch // 2, nn_type='up_' + opt.upsample, norm=norm_layer, activation=activation,
                             dropout=False)
        self.up2_2 = NNBlock(ch // 2, ch // 4, nn_type='up_' + opt.upsample, norm=norm_layer, activation=activation,
                             dropout=False)
        self.out_conv2 = NNBlock(ch // 4, output_nc, nn_type='down_conv7', norm=None, activation=nn.Tanh())

    def forward(self, x1, x2, stage=2):
        y = self.conv1_1(x1)
        y = self.conv1_2(y)
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        h = y
        h = self.up1_1(h)
        self.out1 = self.out_conv1(h)
        if stage == 2:
            z = self.conv2_1(x2)
            z = self.conv2_2(z)
            z = self.conv2_3(z)
            y = torch.cat((y, z), 1)
            y = self.res6(y)
            y = self.res7(y)
            y = self.res8(y)
            y = self.res9(y)
            y = self.up2_1(y)
            y = self.up2_2(y)
            self.out2 = self.out_conv2(y)
            return self.out1, self.out2
        else:
            return self.out1, torch.nn.functional.upsample(self.out1, scale_factor=2)


class Generator_tmp(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', n_downsampling=2, n_upsampling=2, opt=None):
        super(Generator_tmp, self).__init__()
        self.gpu_ids = gpu_ids
        ch = ngf
        activation = nn.LeakyReLU(0.2, True)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        seq = [NNBlock(input_nc, ch, nn_type='conv', norm=norm_layer, activation=activation, dropout=use_dropout)]
        for i in range(n_downsampling):
            seq += [
                NNBlock(ch, ch * 2, nn_type='down_conv', norm=norm_layer, activation=activation, dropout=use_dropout)]
            ch *= 2
        for i in range(n_blocks):
            seq += [ResBlock(ch, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias,
                             shallow=opt.shallow_resblock)]
        for i in range(n_upsampling):
            if opt.upsample == 'conv':
                seq += [NNBlock(ch, ch // 2, nn_type='up_deconv', norm=norm_layer, activation=activation,
                                dropout=use_dropout)]
            if opt.upsample == 'subpix':
                seq += [NNBlock(ch, ch // 2, nn_type='up_subpix', norm=norm_layer, activation=activation,
                                dropout=use_dropout)]
            if opt.upsample == 'scale':
                seq += [NNBlock(ch, ch // 2, nn_type='up_scale', norm=norm_layer, activation=activation,
                                dropout=use_dropout)]
            ch = ch // 2
        seq += [NNBlock(ch, output_nc, kw=9, nn_type='conv', norm=norm_layer, activation=nn.Tanh())]
        self.model = nn.Sequential(*seq)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, gpu_ids=[], opt=None):
        super(Discriminator, self).__init__()
        self.gpu_ids = gpu_ids
        activation = nn.LeakyReLU(0.2, True)
        ch = ndf
        noise = opt.use_noise

        # seq = [NNBlock(input_nc, ndf, nn_type='down_conv', norm=None, activation=activation, dropout=False, noise=noise)]
        # for i in range(1, n_layers):
        #    seq += [NNBlock(ch, ch*2, nn_type='down_conv', norm=None, activation=activation, dropout=False, noise=noise),]
        #    ch*=2
        # seq += [NNBlock(ch, 1, nn_type='conv', norm=None, activation=nn.Sigmoid() if opt.no_lsgan else None, dropout=False, noise=noise),]

        seq = [NNBlock(input_nc, ndf, nn_type='down_conv', norm=None, activation=None, dropout=False, noise=noise)]
        for i in range(1, n_layers):
            seq += [DownResBlock(ch, ch * 2, norm=None, activation=activation, dropout=False, noise=noise), ]
            ch *= 2
        if opt.one_out:
            seq += [NNBlock(ch * opt.feat_len_D, 1, nn_type='linear', norm=None,
                            activation=nn.Sigmoid() if opt.no_lsgan else None), ]
        else:
            seq += [activation,
                    NNBlock(ch, 1, nn_type='conv', norm=None, activation=nn.Sigmoid() if opt.no_lsgan else None,
                            dropout=False, noise=noise), ]

        self.model = nn.Sequential(*seq)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# }}} </editor-fold>

# {{{ <editor-fold desc="Test">

class Bottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, expansion=4, **kwargs):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        # assert outplanes % self.expansion == 0
        interplanes = int(outplanes // self.expansion)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(interplanes)
        self.conv2 = nn.Conv2d(interplanes, interplanes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interplanes)
        self.conv3 = nn.Conv2d(interplanes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or inplanes != outplanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Hourglass_residual(nn.Module):
    def __init__(self, n, f, bn=None, increase=128):
        super(Hourglass_residual, self).__init__()
        nf = f + increase
        self.up1 = Bottleneck(f, f, expansion=2)
        # Lower branch
        self.pool1 = nn.AvgPool2d(2)
        self.low1 = Bottleneck(f, nf, expansion=2)
        # Recursive hourglass
        if n > 1:
            self.low2 = Hourglass_residual(n - 1, nf, bn=bn)
        else:
            self.low2 = Bottleneck(nf, nf, expansion=2)
        self.low3 = Bottleneck(nf, f, expansion=2)
        self.up2 = nn.Upsample(scale_factor=2)
        if f == 3:
            self.outtanh = nn.Tanh()
        else:
            self.outtanh = None

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        if self.outtanh is not None:
            return self.outtanh(up1 + up2)
        else:
            return up1 + up2

def define_F(opt, use_bn=False):
    gpu_ids = opt.gpu_ids
 #   import ipdb; ipdb.set_trace()
    device = torch.device('cuda:%s'%(gpu_ids[0]))
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, \
        use_input_norm=True, device=device)
    # netF = arch.ResNet101FeatureExtractor(use_input_norm=True, device=device)
    if gpu_ids:
        netF = nn.DataParallel(netF)
    netF.eval()  # No need to train
    return netF

# }}} </editor-fold>
