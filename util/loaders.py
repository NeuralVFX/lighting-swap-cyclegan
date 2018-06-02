#LOADERS FOR TRAINING AND TESTING#
import numpy as np
import glob
import os
import random
import copy
import pickle

from tqdm import tqdm
import torchvision.models as models
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
import torch
from torch.utils.data import *


def create_content_model():
    # Create Resnet and chop it to use for content similarity measurment #
    model = models.resnet34(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    classifier_map = list(model.children())
    classifier_map.pop()
    new_map = classifier_map
    new_classifier = nn.Sequential(*new_map)
    extractor = copy.deepcopy(new_classifier)
    extractor.cuda()
    extractor.eval()
    return extractor


def make_content_dict(path_list,input_res = 270):
    # loop through all images provided and fetch content vector #
    scaler = transforms.Resize((224, 224))
    crop = transforms.CenterCrop(input_res)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    extractor = create_content_model()
    content_list = []
    for i in tqdm(path_list):
        try:
            img = Image.open(i)
            normalized_img = Variable(normalize(to_tensor(scaler(crop(img)))).unsqueeze(0)).cuda()
            content = extractor(normalized_img)
            content_list.append(content.view(-1))
        except Exception as e:
            print(e, i)
    del extractor
    return torch.stack(content_list)


class ContentSimilarLoader(Dataset):
    # Loader for training, serves images from each dataset which appear similar, creates cache of most similar images #
    def __init__(self, path_a, path_b, transform, cache=False, cache_file=False, close=30, input_res=270, output_res=128):
        self.input_res = input_res
        self.output_res = output_res
        self.transform = transform
        self.close = close
        print(f'Similarity Distance:{self.close}')
        self.path_list_a = sorted(glob.glob(f'{path_a}/*.*'))
        self.path_list_b = sorted(glob.glob(f'{path_b}/*.*'))
        self.cache_loaded = False
        if cache:
            if os.path.exists(cache_file):
                print('Loaded Existing Content Cache')
                with (open(cache_file, "rb")) as openfile:
                    while True:
                        try:
                            self.similar_lookup = pickle.load(openfile)
                            self.cache_loaded = True
                            print('Cache Loaded')
                        except EOFError:
                            break

        if not self.cache_loaded:
            self.similar_lookup = {'A_sim': [], 'B_sim': []}
            print('Extracting Content')
            self.content_dict_a = make_content_dict(self.path_list_a)
            self.content_dict_b = make_content_dict(self.path_list_b)

            print('Building Lookup')
            for num in tqdm(range(self.content_dict_a.shape[0])):
                chunk = self.content_dict_a[num].unsqueeze(0)
                content_matrix = F.cosine_similarity(chunk.view(-1, 512), self.content_dict_b.view(-1, 512), 1)
                most_similar = sorted(range(content_matrix.shape[0]), key=lambda a: float(content_matrix[a]),
                                      reverse=True)
                self.similar_lookup['A_sim'].append(most_similar)

            for num in tqdm(range(self.content_dict_b.shape[0])):
                chunk = self.content_dict_b[num].unsqueeze(0)
                content_matrix = F.cosine_similarity(chunk.view(-1, 512), self.content_dict_a.view(-1, 512), 1)
                most_similar = sorted(range(content_matrix.shape[0]), key=lambda a: float(content_matrix[a]),
                                      reverse=True)
                self.similar_lookup['B_sim'].append(most_similar)
            if cache_file:
                with open(cache_file, "wb") as f:
                    pickle.dump(self.similar_lookup, f, pickle.HIGHEST_PROTOCOL)

        print('Initialized')

    def transform_set(self, image_a, image_b):
        # adding some slight randomization to the images, yet matching between pairs #
        mult = (random.random() * 1.5 + 1)

        data_transforms = transforms.Compose([
            transforms.RandomRotation(6 * mult, resample=Image.BICUBIC),
            # center crop by an amount that never shows black edges on rotate image #
            transforms.CenterCrop(self.input_res - ((.074*self.input_res) * mult)),
            transforms.RandomResizedCrop(self.output_res, scale=(.9, 1), ratio=(1, 1), interpolation=2)])

        seed = random.randint(0, 2 ** 32)
        np.random.seed(seed)
        image_a = data_transforms(image_a)
        np.random.seed(seed)
        image_b = data_transforms(image_b)
        return image_a, image_b

    def __getitem__(self, index):
        # because each list omits certain samples from the other side, we use both lists put together #
        if index > len(self.path_list_a) - 1:
            index -= len(self.path_list_a)
            sim_index = random.choice(self.similar_lookup['B_sim'][index][:self.close])
            image_path_sim_a = self.path_list_a[sim_index]
            image_path_sim_b = self.path_list_b[index]
        else:
            sim_index = random.choice(self.similar_lookup['A_sim'][index][:self.close])
            image_path_sim_a = self.path_list_a[index]
            image_path_sim_b = self.path_list_b[sim_index]

        image_a = Image.open(image_path_sim_a)
        image_b = Image.open(image_path_sim_b)

        image_a, image_b = self.transform_set(image_a, image_b)

        tensor_a = self.transform(image_a)
        tensor_b = self.transform(image_b)

        return (tensor_a, tensor_b)

    def __len__(self):
        return len(self.path_list_a) + len(self.path_list_b)


class NormalLoader(Dataset):
    # This loaded is used at test time #
    def __init__(self, path_a, transform, input_res=270, output_res=128):
        self.transform = transform
        self.path_list_a = sorted(glob.glob(f'{path_a}/*.*'))
        self.input_res = input_res
        self.output_res = output_res

    def transform_set(self, image_a):
        data_transforms = transforms.Compose([
            # roughly matching crop of training loader for consistency #
            transforms.CenterCrop(self.input_res*.95),
            transforms.Resize(self.output_res)])

        image_a = data_transforms(image_a)
        return image_a

    def __getitem__(self, index):
        image_path_sim_a = self.path_list_a[index]
        image_a = Image.open(image_path_sim_a)
        image_a = self.transform_set(image_a)
        tensor_a = self.transform(image_a)

        return tensor_a

    def __len__(self):
        return len(self.path_list_a)


def data_load(path_a, path_b, transform, batch_size, shuffle=False, cache=False, cache_file=False, close=30,
              input_res=270, output_res=128):
    # Wrapper for content similar loader #
    dataset = ContentSimilarLoader(path_a, path_b, transform, cache=cache, cache_file=cache_file, close=close,
                                   input_res = input_res, output_res =output_res)
    datalen = dataset.__len__()
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=shuffle), datalen


def data_load_preview(path_a, transform, batch_size, shuffle=False, input_res=270, output_res=128):
    # Wrapper for nornal loader #
    dataset = NormalLoader(path_a, transform, input_res = input_res, output_res=output_res)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=shuffle)
