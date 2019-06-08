import torch.utils.data as data
from PIL import Image, ImageOps
import numpy as np
import torchvision.transforms as transforms
import utils.util as util

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

    def tick(self):
        pass

def get_transform(opt, data='-', resample=Image.BICUBIC):
    transform_list = []
    if data == 'C':
	fineSize = opt.CfineSize
	loadSize = opt.CloadSize
    else:
  	fineSize = opt.fineSize
	loadSize = opt.loadSize
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [loadSize, loadSize]
        transform_list.append(transforms.Resize(osize, resample))
        transform_list.append(transforms.RandomCrop(fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, fineSize, resample)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, loadSize, resample)))
        transform_list.append(transforms.RandomCrop(fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    if data in opt.equalize_input:
        transform_list.append(util.hisEqulColor)

    # if opt.isTrain and (data in opt.enh_aug):
    #         transform_list += [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0)]


    if opt.isTrain and (data in opt.aff_aug):
        # import torchsample
        # transform_list += [torchsample.transforms.RandomAffine(rotation_range=0,
        #                                                        translation_range=(0.1, 0.1),
        #                                                        shear_range=0,
        #                                                        zoom_range=(0.9, 1.1))]
        ## TODO fix import error
    	import imgaug as ia
        from imgaug import augmenters as iaa
        to_numpy = lambda img: np.array(img)[np.newaxis,::]
        to_img = lambda arr:Image.fromarray(arr[0])
        aff = iaa.Sequential([
                iaa.Sometimes(0.5,
                   iaa.Affine(
                       scale={"x": (1-opt.scale, 1+opt.scale ), "y": (1-opt.scale, 1+opt.scale)},
                       translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                       mode=["edge"]
                    )
                )])
   	transform_list += [to_numpy, aff.augment_images, to_img]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_width(img, target_width, resample=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), resample)
