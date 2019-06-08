import random
import glob
import os
from PIL import Image
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from utils import util
import align as align
from models import networks
from options.test_options import TestOptions

def get_trans(crop=None, size=None):
    transform_list_show = []
    transform_list = []
    loadSize = fineSize = 128
    if crop is not None:
        loadSize = crop
    if size is not None:
        fineSize = size
    osize = [loadSize, loadSize]
    transform_list_show += [transforms.Scale(osize, Image.BICUBIC),
                       transforms.RandomCrop(fineSize),
                       transforms.ToTensor()]
    img_show = transforms.Compose(transform_list_show)
    transform_list = transform_list_show + [transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
    img_pre = transforms.Compose(transform_list)
    return img_show, img_pre    

def get_idtrans(crop=None, size=None):
    transform_list_show = []
    transform_list = []
    transform_list_show += [transforms.ToTensor()]
    img_show = transforms.Compose(transform_list_show)
    transform_list = transform_list_show + [transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
    img_pre = transforms.Compose(transform_list)
    return img_show, img_pre   

def to_input(img, trans, size):
    img = trans(img)
    input_ = img.view(-1, 3, size, size)
    real = Variable(input_.cuda())
    return real

def test_img(img, net, trans, size, eval_mode=True, bn_eval=True, drop_eval=True):
    img = trans(img)
    #input_ = img.view(-1, 3, size, size)
    input_ = img.unsqueeze(0)
    real = Variable(input_.cuda())
    util.set_eval(net, bn_eval, drop_eval)
    with torch.no_grad():
        fake = net.forward(real)
    return fake

def get_stack_nets(path, which_epoch = 'latest'):
    gpu_ids = [0]
    Tensor = torch.cuda.FloatTensor
    opt = util.load_opt(path)
    netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                            opt.ngf, opt.which_model_netG, opt.norm, False, opt.init_type, 
                            gpu_ids, n_upsampling=3, opt=opt)
    netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                            opt.ngf, opt.which_model_netG, opt.norm, False, opt.init_type, 
                            gpu_ids, n_upsampling=3, opt=opt)
    netG_A_pre = networks.define_G(opt.input_nc, opt.output_nc,
                            opt.ngf, opt.which_model_netG, opt.norm, False, opt.init_type, 
                            gpu_ids, n_downsampling=3, opt=opt)
    netG_B_pre = networks.define_G(opt.output_nc, opt.input_nc,
                            opt.ngf, opt.which_model_netG, opt.norm, False, opt.init_type, 
                            gpu_ids, n_downsampling=3, opt=opt)

    load_network_with_path(netG_A,  'G_A', which_epoch, path)
    load_network_with_path(netG_B,  'G_B', which_epoch, path)
    load_network_with_path(netG_A_pre,  'G_A_pre', which_epoch, path)
    load_network_with_path(netG_B_pre,  'G_B_pre', which_epoch, path)
        
    netG_A.cuda()
    netG_B.cuda()
    netG_A_pre.cuda()
    netG_B_pre.cuda()
    
    return {'A':netG_A, 'B':netG_B, 'A_pre':netG_A_pre, 'B_pre':netG_B_pre}

def get_stack_nets(path, which_epoch = 'latest', skip=True):
    gpu_ids = [0]
    Tensor = torch.cuda.FloatTensor
    opt = util.load_opt(path)
    opt.max_ngf = 256
    netG_A = networks.define_G(opt.input_nc if not skip else opt.input_nc * 2, opt.output_nc,
                            opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, 
                            gpu_ids, n_upsampling=opt.n_upsample, opt=opt)
    netG_B = networks.define_G(opt.input_nc if not skip else opt.input_nc * 2, opt.input_nc,
                            opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, 
                            gpu_ids, n_upsampling=opt.n_upsample, opt=opt)
    if 'pre_path' in vars(opt).keys():
        opt_pre = load_opt(opt.pre_path)
        netG_A_pre = networks.define_G(opt.input_nc, opt.output_nc,
                                opt_pre.ngf, opt_pre.which_model_netG, opt_pre.norm, not opt_pre.no_dropout, opt_pre.init_type, 
                                gpu_ids, n_downsampling=3, opt=opt_pre)
        netG_B_pre = networks.define_G(opt.output_nc, opt.input_nc,
                                opt_pre.ngf, opt.which_model_netG, opt_pre.norm, not opt_pre.no_dropout, opt_pre.init_type, 
                                gpu_ids, n_downsampling=3, opt=opt_pre)
    else:
        opt.pre_path = ''
        netG_A_pre = networks.define_G(opt.input_nc, opt.output_nc,
                            opt.ngf, opt.which_model_netG, opt.norm, False, opt.init_type, 
                            gpu_ids, n_downsampling=3, opt=opt)
        netG_B_pre = networks.define_G(opt.output_nc, opt.input_nc,
                            opt.ngf, opt.which_model_netG, opt.norm, False, opt.init_type, 
                            gpu_ids, n_downsampling=3, opt=opt)

    load_network_with_path(netG_A,  'G_A', which_epoch, path)
    load_network_with_path(netG_B,  'G_B', which_epoch, path)
    load_network_with_path(netG_A_pre,  'G_A_pre', which_epoch, path if opt.pre_path == '' else opt.pre_path)
    load_network_with_path(netG_B_pre,  'G_B_pre', which_epoch, path if opt.pre_path == '' else opt.pre_path)
    netG_A.cuda()
    netG_B.cuda()
    netG_A_pre.cuda()
    netG_B_pre.cuda()
    return {'A':netG_A, 'B':netG_B, 'A_pre':netG_A_pre, 'B_pre':netG_B_pre}

def get_nets(path, which_epoch = 'latest', def_opt=None):
    gpu_ids = [0]
    Tensor = torch.cuda.FloatTensor
    opt = util.load_opt(path, def_opt)
    # assume caffe style model
    opt.not_caffe = False
    netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                  opt.ngf, opt.which_model_netG,
                                  opt.norm,  not opt.no_dropout, opt.init_type,
                                  gpu_ids, opt=opt)
    util.load_network_with_path(netG_A, 'G_A', path)
    
    netG_B = networks.define_G(opt.input_nc, opt.output_nc,
                                  opt.ngf, opt.which_model_netG,
                                  opt.norm,  not opt.no_dropout, opt.init_type,
                                  gpu_ids, opt=opt)
    util.load_network_with_path(netG_B, 'G_B', path)
    netG_A.cuda()
    netG_B.cuda()
    return {'A':netG_A, 'B':netG_B}

def get_nets_dis(path, which_epoch = 'latest'):
    gpu_ids = [0]
    Tensor = torch.cuda.FloatTensor
    opt = util.load_opt(path)
    # assume caffe style model
    opt.caffe=False
    netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                    opt.which_model_netD,
                                    opt.n_layers_D, opt.norm, False, opt.init_type, gpu_ids, opt=opt)
    netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                    opt.which_model_netD,
                                    opt.n_layers_D, opt.norm, False, opt.init_type, gpu_ids, opt=opt)
    load_network_with_path(netD_A, 'D_A', which_epoch, path)
    load_network_with_path(netD_B, 'D_B', which_epoch, path)
    netD_A.cuda()
    netD_B.cuda()
    return {'A':netD_A, 'B':netD_B}


def show(img, show=True):
    npimg = img.numpy()
    npimg = ((npimg / 2.0) + 0.5) * 255.0
    npimg = npimg.astype(np.uint8)
    if show:
        plt.figure(figsize = (20,20))
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    return Image.fromarray(npimg.transpose((1,2,0)))

def resize_to_size(tensor, out_size=128):
    img = Image.fromarray(util.util.tensor2im(tensor)).resize((out_size, out_size))
    return img


from PIL import ImageOps
from IPython.html.widgets import interact
from PIL import Image
def resize_plot(x=0.37, d=0.87, rot=1.0):
    B_ = Image.open(B_p).convert('RGB')#.crop((1445,0,3666,2028))
    max_base_width = 512
    wpercent = (max_base_width/float(B_.size[0]))
    hsize = int((float(B_.size[1])*float(wpercent)))
    B_1 = B_.resize((max_base_width,hsize), Image.BICUBIC).rotate(0)
    B_2 = B_1
    if rot!= 0:
        B_2_orig = B_2
        np_im = np.array(B_2)
        pad_type = 'edge'
        pad_size = max(B_2.size[0], B_2.size[1]) / 2
        tmp_im = Image.fromarray(np.rot90(np.array([np.pad(np_im[:,:,0], pad_size, pad_type), \
                                  np.pad(np_im[:,:,1], pad_size, pad_type), \
                                  np.pad(np_im[:,:,2], pad_size, pad_type)]).T, 3))
        B_2 = ImageOps.mirror(tmp_im)
        B_2 = B_2.rotate(rot, Image.BICUBIC)
        B_2 = B_2.crop((pad_size, pad_size, pad_size+B_2_orig.size[0], pad_size+B_2_orig.size[1]))
    B_2 = B_2.resize((int((B_1.size[0])*x),int((B_1.size[1])*(x*d))), Image.BICUBIC)
#     B_2_pad = Image.new('RGB',(B_2.size[0]+60, B_2.size[1]+60), (0,0,0))
#     np_im = np.array(B_2)
#     pad_type = 'mean'
#     pad_size = 30
#     tmp_im = Image.fromarray(np.rot90(np.array([np.pad(np_im[:,:,0], pad_size, pad_type), \
#                               np.pad(np_im[:,:,1], pad_size, pad_type), \
#                               np.pad(np_im[:,:,2], pad_size, pad_type)]).T, 3))
#     B_2_pad = ImageOps.mirror(tmp_im)
#     B_2 = B_2_pad
#     print B_2_pad.size, B_2.size
    fake_A_1 = test_img(B_2, net_2['B'], img_id_pre, 128, eval_mode=True)
    B_2_show = Image.new('RGB', (fake_A_1.shape[-1], fake_A_1.shape[-2]), (0,0,0))
    print B_2.size, B_2_show.size
    B_2_show.paste(B_2, (0,0))#B_2.getbbox())
    show(torchvision.utils.make_grid([img_id_pre(B_2_show),fake_A_1.data[0].cpu()]))

# interact(resize_plot, x=(0,1,0.01), d=(0.5,1.2,0.01), rot=(-30,30,1))

def resize_plot2(x=0.27, d=0.0):
    B_ = Image.open(B_p).convert('RGB').crop((129,0,479,464))
    base_width = 512
    wpercent = (base_width/float(B_.size[0]))
    hsize = int((float(B_.size[1])*float(wpercent)))
    B_1 = B_.resize((base_width,hsize), Image.BICUBIC)
    B_2 = B_1.resize((int((B_1.size[0])*x),int((B_1.size[1])*(x+d))), Image.ANTIALIAS)
    fake_A_1 = test_img(B_2, net_2['B'], img_id_pre, 128, eval_mode=True)
    B_2_show = Image.new('RGB', (fake_A_1.shape[-1], fake_A_1.shape[-2]), (0,0,0))
    print B_2.size, B_2_show.size
    B_2_show.paste(B_2, B_2.getbbox())
    #print net_1_d['B'].forward(fake_A_1).mean().data 
    try:
        show(torchvision.utils.make_grid([img_id_pre(B_2_show),fake_A_1.data[0].cpu()]))
    except Exception:
        show(torchvision.utils.make_grid([img_id_pre(B_2_show)]))
        show(torchvision.utils.make_grid([fake_A_1.data[0].cpu()]))

# interact(resize_plot2, x=(0,1,0.01), d=(-0.5,0.5,0.01))
