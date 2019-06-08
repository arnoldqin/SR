import random
import numpy as np
import torch
from torch.autograd import Variable
class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def get_variable_data(var):
        pass

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        if isinstance(images, (list, tuple)):
            # asumming batch size 1
            input_images = [i.detach() for i in images]
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(input_images)
                return_images = input_images
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id]
                    self.images[random_id] = input_images
                    return_images = tmp
                else:
                    return_images = input_images
        else:
            for image in images.data:
                image = torch.unsqueeze(image, 0)
                if self.num_imgs < self.pool_size:
                    self.num_imgs = self.num_imgs + 1
                    self.images.append(image)
                    return_images.append(image)
                else:
                    p = random.uniform(0, 1)
                    if p > 0.5:
                        random_id = random.randint(0, self.pool_size-1)
                        tmp = self.images[random_id].clone()
                        self.images[random_id] = image
                        return_images.append(tmp)
                    else:
                        return_images.append(image)
            return_images = Variable(torch.cat(return_images, 0))
        return return_images
