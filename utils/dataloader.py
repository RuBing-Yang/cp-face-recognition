# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import bisect
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from PIL import Image

from typing import TypeVar, Generic, Iterable, Iterator, Sequence, List, Optional, Tuple


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

def train_data_loader(data_path, img_size, use_augment=False):
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    if use_augment:\
        # TODO: augment parameters need to be tuned
        data_transforms = transforms.Compose([
            transforms.RandomOrder([
                transforms.RandomApply([transforms.ColorJitter(contrast=0.5)], .5),
                transforms.Compose([
                    transforms.RandomApply([transforms.ColorJitter(saturation=0.5)], .5),
                    transforms.RandomApply([transforms.ColorJitter(hue=0.1)], .5),
                ])
            ]),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.125)], .5),
            transforms.RandomApply([transforms.RandomRotation(15)], .5),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize    
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    faces_dataset = ImageFolder(data_path, data_transforms,
        is_valid_file=lambda path: os.path.basename(path)[0].lower() == 'p'
    )

    ori_cartoon_dataset = ImageFolder(data_path, data_transforms,
        is_valid_file=lambda path: os.path.basename(path)[0].lower() == 'c'
    )

    grey_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    grey_cartoon_dataset = ImageFolder(data_path, data_transforms,
        is_valid_file=lambda path: os.path.basename(path)[0].lower() == 'c'
    )

    cartoon_dataset = torch.utils.data.ConcatDataset(
        [ori_cartoon_dataset, grey_cartoon_dataset]
    )

    return AlignedDataset([faces_dataset, cartoon_dataset])


def test_data_loader(data_path):

    # return full path
    queries_path = [os.path.join(data_path, 'query', path) for path in os.listdir(os.path.join(data_path, 'query'))]
    references_path = [os.path.join(data_path, 'reference', path) for path in
                       os.listdir(os.path.join(data_path, 'reference'))]

    return queries_path, references_path


def test_data_generator(data_path, img_size):
    img_size = (img_size, img_size)
    data_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_image_dataset = TestDataset(data_path, data_transforms)

    return test_image_dataset


class AlignedDataset(Dataset):
    """
    Requires: 
        - every sub-dataset share the same labels 
        - every subdataset is instance of `torchvsion.VisionDataset` which is not assigned below 
    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    def cumsum(self):
        r, s = [], 0
        for target in self.targets_set:
            l = max(map(lambda x: len(x[target]), self.targets_indices_map))
            r.append(l + s)
            s += l
        return r

    def targets_to_indices(self, dataset):
        targets_list = []
        for _, target in dataset:
            targets_list.append(target)
        targets = torch.LongTensor(targets_list).numpy()
        
        return {target: np.where(targets == target)[0]
                    for target in self.targets_set}
        
    
    def __init__(self, datasets: Iterable[DatasetFolder]) -> None:
        super(AlignedDataset, self).__init__()
        assert len(datasets) == 2, \
               'currently only support 2 datasets to align'
        
        assert len(datasets) > 0, \
               'datasets should not be an empty iterable'  # type: ignore
        self.datasets = list(datasets)
        
        assert hasattr(self.datasets[0], 'targets'), \
               "first dataset shold have attiribute 'targets'"
        self.targets_set = set(self.datasets[0].targets)

        for d in self.datasets:
            assert not isinstance(d, IterableDataset), \
                   "AlignedDataset not support IterableDataset"
        
        self.targets_indices_map = \
            [self.targets_to_indices(d) for d in self.datasets]

        self.cumulative_sizes = self.cumsum()


    def __getitem__(self, index) -> Tuple[T_co]:
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index
        target = bisect.bisect_right(self.cumulative_sizes, index)
        if target == 0:
            sample_idx = index
        else:
            sample_idx = index - self.cumulative_sizes[target - 1]
        
        item = []
        for d, m in zip(self.datasets, self.targets_indices_map):
            l = len(m[target])
            item.append(d[m[target][sample_idx % l]][0])
        item.append(target)

        return tuple(item)

    def __len__(self):
        return self.cumulative_sizes[-1]


class TestDataset(Dataset):
    def __init__(self, img_path_list, transform=None):
        self.img_path_list = img_path_list
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img_path, img

    def __len__(self):
        return len(self.img_path_list)


if __name__ == '__main__':
    query, refer = test_data_loader('./')
    print(query)
    print(refer)
