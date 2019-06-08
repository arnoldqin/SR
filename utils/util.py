from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import argparse
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

def mse_loss(input, target):
    return torch.sum((input - target)**2) / input.data.nelement()

def l1_loss(input, target):
    return torch.sum(torch.abs(input - target)) / input.data.nelement()

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, type='small'):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.clip(image_numpy,-1,1)
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    if type == 'small':
	image_numpy = np.pad(image_numpy,((64,64),(64,64),(0,0)),'constant',constant_values = (255,255))
    return image_numpy.astype(imtype)

def get_params(net):
    return [(name, param.clone().cpu().data.numpy()) for name, param in net.named_parameters()]

def get_grads(net, ret_type='hist'):
    if ret_type == 'hist':
        return [(name+'.grad', param.grad.clone().cpu().data.numpy()) for name, param in net.named_parameters()]
    elif ret_type == 'sum':
        g_list = [param.grad.norm() for name, param in net.named_parameters()]
        ret = 0.0
        for g in g_list:
            ret += g
        return ret / len(g_list)

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_opt(path, def_opt=None, overwrite={}):
        ret = argparse.Namespace()
        var = vars(ret)
        input_list = [i.strip() for i in open(os.path.join(path,'opt.txt'))]
        for line in input_list:
            if ':' in line:
                k, v = line.split(':')
                try:
                    var[k]=eval(v)
                except:
                    var[k]=v if len(v) < 1 else v[1:]
        if def_opt is not None:
            var_def = vars(def_opt)
            for k in var_def.keys():
                if k not in var.keys():
                    var[k] = var_def[k]
        for k in overwrite:
            var[k] = overwrite[k]
        if ret.load_pre and ret.pre_path != '':
            ret.pre = load_opt(ret.pre_path, def_opt)
        return ret

def load_network_with_path(network, network_label, path, epoch_label='latest'):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(path, save_filename)
    load_dict = torch.load(save_path)
    network_dict = network.state_dict()
    for key, var in load_dict.items():
        try:
            if len(var.shape) > 1:
                network_dict[key][0:var.shape[0], 0:var.shape[1]].copy_(var)
            else:
                network_dict[key][0:var.shape[0]].copy_(var)
        except Exception:
            print('Unable to load layer in :', network_label, key)

    
def print_var_memory():
    # prints currently alive Tensors and Variables
    import gc
    for obj in gc.get_objects():
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size())

def set_eval(net, bn=False, drop=True):
    if isinstance(net, torch.nn.Dropout):
        net.training = not drop
    elif isinstance(net, torch.nn.BatchNorm2d) or isinstance(net, torch.nn.InstanceNorm2d):
        net.training = not bn
    else:
        net.training = False
    for i in net.children():
        set_eval(i, bn, drop)

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

        
def parallel_func(func, arg_list, num_proc=None):
    from multiprocessing import Pool
    pool=Pool(processes=num_proc) 
    pool.map(func,arg_list)
    pool.close()
    pool.join()
    
def hisEqulColor(img):
    img_np = np.array(img)
    img_cv = img_np[:,:,::-1]
    ycrcb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    img_cv = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    img_np = img_cv[:,:,::-1]
    img = Image.fromarray(img_np)
    return img
