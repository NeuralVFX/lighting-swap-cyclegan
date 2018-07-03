# HELPER UTILITIES
import random

import torch
from torch.autograd import Variable
from torchvision import transforms
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from util import loaders as load


############################################################################
# Helper Utilities
############################################################################

def weights_init_normal(m):
    # Set initial state of weights
    classname = m.__class__.__name__
    if 'ConvTrans' == classname:
        pass
    elif 'Conv2d' in classname or 'Linear' in classname or 'ConvTrans' in classname:
        nn.init.normal(m.weight.data, 0, .02)


def mft(tensor):
    # Return mean float tensor
    return torch.mean(torch.FloatTensor(tensor))


def normalize_img(x, cpu=False):
    # Reverse Image Normalization
    if cpu:
        x = x.cpu().data
    return (x.numpy().transpose(1, 2, 0) + 1) / 2


def show_test(g, g_a, params, save=False):
    # Use IDs provided by user to index into non-shufled data loader and chech the same images each time
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))])
    ids_a = params['ids_a']
    ids_b = params['ids_b']
    input_res = params["img_input_size"]
    output_res = params["img_output_size"]

    test_loader_a = load.data_load_preview(f'data/{params["dataset"]}/{params["test_folder"]}/{params["A"]}', transform,
                                           1, shuffle=False, input_res=input_res, output_res = output_res)
    test_loader_b = load.data_load_preview(f'data/{params["dataset"]}/{params["test_folder"]}/{params["B"]}', transform,
                                           1, shuffle=False, input_res=input_res, output_res = output_res)

    # show and save
    g.eval()
    g_a.eval()
    image_grid_len = len(ids_a) + len(ids_b)
    fig, ax = plt.subplots(image_grid_len, 2, figsize=(6, 12))

    count = 0
    for idx, real_a in enumerate(test_loader_a):
        if idx in ids_a:
            real_a = Variable(real_a.cuda())
            test = g(real_a)

            ax[count, 0].cla()
            ax[count, 0].imshow(normalize_img(real_a[0], cpu=True))
            ax[count, 1].cla()
            ax[count, 1].imshow(normalize_img(test[0], cpu=True))
            count += 1
    for idx, real_b in enumerate(test_loader_b):
        if idx in ids_b:
            real_b = Variable(real_b.cuda())
            test = g_a(real_b)
            ax[count, 0].cla()
            ax[count, 0].imshow(normalize_img(real_b[0], cpu=True))
            ax[count, 1].cla()
            ax[count, 1].imshow(normalize_img(test[0], cpu=True))
            count += 1
    g.train()
    g_a.train()
    if save:
        plt.savefig(save)
    plt.show()
    plt.close(fig)


class ImagePool:
    #Use old images during training: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/75
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > .5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images

