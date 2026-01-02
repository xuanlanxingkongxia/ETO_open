import os
import random

import numpy as np
import torch
import torch.utils.data.distributed
from torch.utils.data import ConcatDataset, DataLoader
import torchvision.transforms as transforms
from datasets.augmentation import (ha_augment_sample, resize_sample,
                                         spatial_augment_sample,
                                         to_tensor_sample)
from datasets.coco import COCOLoader
import glob
from PIL import Image
from utils.utils import Create_pair_list_H

def sample_to_cuda(data):
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        data_cuda = {}
        for key in data.keys():
            data_cuda[key] = sample_to_cuda(data[key])
        return data_cuda
    elif isinstance(data, list):
        data_cuda = []
        for key in data:
            data_cuda.append(sample_to_cuda(key))
        return data_cuda
    else:
        return data.to('cuda')


def image_transforms(shape, jittering):
    def train_transforms(sample):
        sample = resize_sample(sample, image_shape=shape)
        sample = spatial_augment_sample(sample)
        sample = to_tensor_sample(sample)
        sample = ha_augment_sample(sample, jitter_paramters=jittering)
        return sample

    return {'train': train_transforms}


def _set_seeds(seed=42):
    """Set Python random seeding and PyTorch seeds.
    Parameters
    ----------
    seed: int, default: 42
        Random number generator seeds for PyTorch and python
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def Create_new_dataset(path):
    data_path = path + "train2017/"
    transform_path = []
    transform_path.append(path+"train2017_transform1/")
    transform_path.append(path+"train2017_transform2/")
    transform_path.append(path+"train2017_transform3/")
    transform_path.append(path+"train2017_transform4/")
    transform_path.append(path+"train2017_transform5/")
    homography_list = []
    for file_path in glob.glob(data_path + '/*.jpg'):
        filename = os.path.split(file_path)[1]
        img = Image.open(file_path)
        homography = []
        sample = {}
        for i in range(5):
            if img.mode == 'L':
                image_new = Image.new("RGB", img.size)
                image_new.paste(img)
                sample['image'] = image_new
            else:
                sample['image'] = img
            sample = resize_sample(sample, image_shape=(480, 640))
#            sample = spatial_augment_sample(sample)
            sample = to_tensor_sample(sample)
            sample = ha_augment_sample(sample, jitter_paramters=(0, 0, 0, 0))
            target = sample['image_aug'].cpu()
            H = sample['homography'].cpu().numpy()
            toPIL = transforms.ToPILImage()
            pic = toPIL(target)
            pic.save(transform_path[i] + filename)
            homography.append(H)
        homography_list.append(homography)
        np.save(path + "Homograpy_record", homography_list)

# 除了以前的这些东西之外，还需要保存什么东西呢？大概就只有mask是必须保存的，其他没了
def Create_efficient_dataset(path):
    toPIL = transforms.ToPILImage()
    data_path = path + "train2017/"
    transform_path = []
    transform_path.append(path+"train2017_transform1/")
    transform_path.append(path+"train2017_transform2/")
    transform_path.append(path+"train2017_transform3/")
    transform_path.append(path+"train2017_transform4/")
    transform_path.append(path+"train2017_transform5/")
    transform_path.append(path+"train2017/")
    homography_list = []
    homography_load = np.load(path + "Homograpy_record.npy")
    for i in range(homography_load.shape[0]):
        homography_list.append(homography_load[i])
    data_list = np.array(glob.glob("/media/junjieni/B0AA57DDAA579F22/dataset/coco2017/train2017/" + '/*.jpg'))[:500]
    for (k, file_path) in enumerate(data_list):
        print(k)
        filename = os.path.split(file_path)[1]
        img = Image.open(file_path)
        homography = []
        sample = {}
        if img.mode == 'L':
            image_new = Image.new("RGB", img.size)
            image_new.paste(img)
            sample['image'] = image_new
        else:
            sample['image'] = img
        sample = resize_sample(sample, image_shape=(480, 640))
        #            sample = spatial_augment_sample(sample)
        sample = to_tensor_sample(sample)
        select02 = np.where(np.random.rand() > 0.5, 0, 1)
        select13 = np.where(np.random.rand() > 0.5, 2, 3)
        scales02 = np.exp(np.random.rand() * 2 * np.log(2) - np.log(2))
        scales13 = np.exp(np.random.rand() * 2 * np.log(2) - np.log(2))
        hw_ratio02 = np.exp(np.random.rand() * np.log(3))
        hw_ratio13 = np.exp(np.random.rand() * np.log(3))
        perspective_length02 = 2 - np.random.rand() * (1 - 0.3) * 2
        perspective_start02 = np.random.rand() * (2 - perspective_length02)
        perspective_length13 = 2 - np.random.rand() * (1 - 0.3) * 2
        perspective_start13 = np.random.rand() * (2 - perspective_length13)
        for i in range(5):
            if i == 0:
                sample = ha_augment_sample(sample, scales=scales02, select=select02,
                    mode=0, data_path=data_list,
                    hw_ratio=hw_ratio02, perspective_start=perspective_start02, perspective_length=perspective_length02)
            if i == 1:
                sample = ha_augment_sample(sample, scales=scales13, select=select13,
                    mode=0, data_path=data_list,
                    hw_ratio=hw_ratio13, perspective_start=perspective_start13, perspective_length=perspective_length13)
            if i == 2:
                sample = ha_augment_sample(sample, scales=scales02, select=select02,
                    mode=2, data_path=data_list,
                    hw_ratio=hw_ratio02, perspective_start=perspective_start02, perspective_length=perspective_length02)
            if i == 3:
                sample = ha_augment_sample(sample, scales=scales13, select=select13,
                    mode=3, data_path=data_list,
                    hw_ratio=hw_ratio13, perspective_start=perspective_start13, perspective_length=perspective_length13)
            if i == 4:
                sample = ha_augment_sample(sample, data_path=data_list, mode=4)
            target = sample['image_aug'].cpu()
            H = sample['homography'].cpu().numpy()
            pic = toPIL(target)
            pic.save(transform_path[i] + filename)
            np.save(transform_path[i] + filename, sample['mask'])
            homography.append(H)
        source = sample['image'].cpu()
        pic = toPIL(source)
        pic.save(transform_path[5] + filename)
        homography_list.append(homography)
        np.save(path + "Homograpy_record2", homography_list)

def Create_test_dataset(path):
    toPIL = transforms.ToPILImage()
    data_path = path + "train2017/"
    transform_path = []
    transform_path.append(path+"scale_1/")
    transform_path.append(path+"scale_2/")
    transform_path.append(path+"scale_3/")
    transform_path.append(path+"scale_4/")
    transform_path.append(path+"scale_5/")
    transform_path.append(path+"train2017/")
    homography_list = []
    # homography_load = np.load(path + "Homograpy_record.npy")
    # for i in range(homography_load.shape[0]):
    #     homography_list.append(homography_load[i])
    data_list = np.array(glob.glob("/media/junjieni/B0AA57DDAA579F22/dataset/coco2017/train2017/" + '/*.jpg'))
    for (k, file_path) in enumerate(data_list):
        print(k)
        filename = os.path.split(file_path)[1]
        img = Image.open(file_path)
        homography = []
        sample = {}
        if img.mode == 'L':
            image_new = Image.new("RGB", img.size)
            image_new.paste(img)
            sample['image'] = image_new
        else:
            sample['image'] = img
        sample = resize_sample(sample, image_shape=(480, 640))
        #            sample = spatial_augment_sample(sample)
        sample = to_tensor_sample(sample)
        select = np.array((np.random.rand() * 4)).astype(int)
        hw_ratio = np.exp(np.random.rand() * np.log(3))
        perspective_length = 2 - np.random.rand() * (1 - 0.3) * 2
        perspective_start = np.random.rand() * (2 - perspective_length)
        for i in range(5):
            sample = ha_augment_sample(sample, scales= i + 1, select=select,
                    mode=0, data_path=data_list,
                    hw_ratio=hw_ratio, perspective_start=perspective_start, perspective_length=perspective_length)
            target = sample['image_aug'].cpu()
            H = sample['homography'].cpu().numpy()
            pic = toPIL(target)
            pic.save(transform_path[i] + filename)
            np.save(transform_path[i] + filename, sample['mask'])
            homography.append(H)
        source = sample['image'].cpu()
        pic = toPIL(source)
        pic.save(transform_path[5] + filename)
        homography_list.append(homography)
        np.save(path + "Homograpy_record2", homography_list)


data_path = "/media/junjieni/B0AA57DDAA579F22/dataset/coco2017_new/"
#Create_pair_list_H(data_path)
# Create_efficient_dataset(data_path)
data_path = "/media/junjieni/B0AA57DDAA579F22/dataset/coco2017_test/"
Create_test_dataset(data_path)