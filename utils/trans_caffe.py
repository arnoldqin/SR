from __future__ import absolute_import
import random
import glob
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from models import networks
import utils.visualize_test as visualize_test
from options.test_options import TestOptions
opt = TestOptions().parse()

from pkgs import pytorch2caffe

net_1 = visualize_test.get_nets('./checkpoints/Get_AisCel_128_4_aff_1_resize_caffe/', def_opt = opt)

m = net_1['B'].cpu()
m.eval()
print(m)

input_var = Variable(torch.rand(1, 3, 128, 128))
output_var = m(input_var)

output_dir = 'caffe'
pytorch2caffe.pytorch2caffe(input_var, output_var, 
              os.path.join(output_dir, 'trans.prototxt'),
              os.path.join(output_dir, 'trans.caffemodel'))

import caffe
import numpy as np
from PIL import Image

ipdb.set_trace()

testimg = Image.open("dataset/test_img/nan1_aligned.jpg")
np_testimg = np.array(testimg.resize((128,128),Image.BILINEAR))[np.newaxis,::].transpose(0,3,1,2)
np_testimg = np_testimg / 255.0
np_testimg = (np_testimg - 0.5) / 0.5

net = caffe.Net(os.path.join(output_dir, 'trans.prototxt'),
                os.path.join(output_dir, 'trans.caffemodel'), 0)

caffeout = net.forward_all(**{"data":np_testimg})['TanhBackward67']
print caffeout, caffeout.shape

th_testimg = Variable(torch.from_numpy(np_testimg)).type(torch.FloatTensor).cuda()
m = net_1['B'].cuda()
torchout = m(th_testimg).data[0]

print torchout, torchout.shape

print "mse: ", ((caffeout[0] -  torchout.cpu().numpy())**2).sum() / (caffeout[0].size)

def toimg(npimg):
    npimg = ((npimg / 2.0) + 0.5) * 255.0
    npimg = npimg.astype(np.uint8)
    return Image.fromarray(npimg.transpose((1,2,0)))

toimg(caffeout[0]).save('./caffe.jpg')
toimg(torchout.cpu().numpy()).save('./torch.jpg')
