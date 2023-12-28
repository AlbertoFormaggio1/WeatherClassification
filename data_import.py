import torch
import torchvision.datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torchvision as tv
from transformers import AutoImageProcessor
from typing import List, Tuple
from pathlib import Path
import os
import numpy as np

NUM_WORKERS = os.cpu_count()

def map_labels(datasets: List[dict], image_processor: AutoImageProcessor, final_labels: List[str], batch_size: int, aug_type: str = None):
    labels_ds = []
    label2id_ds = []
    label_ass_ds = []
    ass_ds_final = []
    for ds in datasets:
        labels_ds.append(ds['classes'])
        label2id_ds.append({c: i for i, c in enumerate(labels_ds[-1])})

        if 'shine' in labels_ds[-1]:
            ass_ds_final.append({'shine': 'clear', 'cloudy': 'fog', 'rain': 'rain', 'sunrise': 'sunrise'})  # MWD ds
        elif 'snow' in labels_ds[-1]:
            ass_ds_final.append(
                {'clear': 'clear', 'fog': 'fog', 'rain': 'rain', 'night': 'night', 'snow': 'snow'})  # ACDC ds
        else:
            ass_ds_final.append({'clear': 'clear', 'fog': 'fog', 'rain': 'rain', 'night': 'night'}) # Uav and syndrone

    # Get the index in the final_labels array corresponding to the label in the current dataset
    for idx, l_ds in enumerate(labels_ds):
        label_ass_ds.append({})
        for label in l_ds:
            true_l = ass_ds_final[idx][label]
            ind = np.where(true_l == final_labels)
            label_ass_ds[-1][label2id_ds[idx][label]] = ind[0][0]

    # replace the old label with the new one
    for idx, ds in enumerate(datasets):
        # replace targets over train and test set (positions 0 and 1 of each dataset)
        for key, split in ds.items():
            if key == 'train' or key == 'test':
                for (old_l, new_l) in label_ass_ds[idx].items():
                    split.targets[split.targets == old_l] = new_l

    return datasets, ass_ds_final


def compute_weights(datasets):
    len_train = [len(ds['train']) for ds in datasets]
    num_samples = sum(len_train)
    weights = torch.tensor(len_train, dtype=torch.float32) / num_samples

    return 1 - weights


def gen_dataloader(datasets, image_processor, weights, ass_ds_final, batch_size, aug_type):
    dataloaders = {}

    # Generate a weighted sampler oversampling those samples belonging to the minority class
    num_samples = sum([len(ds['train']) for ds in datasets])
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)

    ds_to_merge = [ds['train'] for ds in datasets]
    merged_train_ds = torch.utils.data.ConcatDataset(ds_to_merge)

    merged_train_ds, merged_eval_ds = torch.utils.data.random_split(merged_train_ds, [0.8, 0.2])
    pre_train_ds = image_preprocessing(merged_train_ds, image_processor, aug_type, 'train')
    pre_eval_ds = image_preprocessing(merged_eval_ds, image_processor, aug_type, 'eval')

    train_dataloader = DataLoader(pre_train_ds, sampler=sampler, batch_size=batch_size, num_workers=NUM_WORKERS,
                                  drop_last=True)
    eval_dataloader = DataLoader(pre_eval_ds, sampler=sampler, batch_size=batch_size, num_workers=NUM_WORKERS,
                                 drop_last=True)

    dataloaders['train'] = train_dataloader
    dataloaders['eval'] = eval_dataloader
    dataloaders['test'] = []

    for idx, ds in enumerate(datasets):
        pre_test_ds = image_preprocessing(ds['test'], image_processor, aug_type, 'eval')

        test_dataloader = DataLoader(pre_test_ds, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False,
                                     drop_last=True)

        dataloaders['test'].append({'ds': test_dataloader, 'name': ds['name'], 'ass': ass_ds_final[idx]})

    return dataloaders


def get_ds(ds_name: Path):
    train_ds = tv.datasets.ImageFolder(str(ds_name / 'train'))
    test_ds = tv.datasets.ImageFolder(str(ds_name / 'test'))

    classes = train_ds.classes

    return {"train": train_ds, "test": test_ds, "classes": classes, 'name': ds_name}


def image_preprocessing(ds, image_processor: AutoImageProcessor, aug_type: str, type: str):
    # Normalize the features with the mean and std dev used
    normalize = v2.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
        max_size = None
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)
        max_size = image_processor.size.get("longest_edge")

    if(type == 'train'):
        ds_proc = image_train_preprocessing(ds, crop_size, normalize, aug_type)
    else:
        ds_proc = image_test_preprocessing(ds, size, crop_size, normalize)

    return ds_proc


def image_train_preprocessing(train_ds: torchvision.datasets.ImageFolder, crop_size: Tuple[int],
                              normalize: v2.Normalize, aug_type: str):
    if aug_type == None or aug_type == 'base':
        _train_transform = v2.Compose([v2.RandomResizedCrop(crop_size),
                                       v2.RandomHorizontalFlip(),
                                       v2.ToTensor(),
                                       normalize])
    elif aug_type == 'geometric_simple':
        _train_transform = v2.Compose([v2.RandomResizedCrop(crop_size),
                                       v2.RandomHorizontalFlip(),
                                       v2.RandomRotation(degrees=20),
                                       lambda x: v2.functional.adjust_brightness(x, 0.2),
                                       v2.ToTensor(),
                                       normalize
                                       ])
    elif aug_type == 'geometric_simple_v':
        _train_transform = v2.Compose([v2.RandomResizedCrop(crop_size),
                                       v2.RandomHorizontalFlip(),
                                       v2.RandomVerticalFlip(p=0.2),
                                       v2.RandomRotation(degrees=20),
                                       lambda x: v2.functional.adjust_brightness(x, 0.2),
                                       v2.ToTensor(),
                                       normalize
                                       ])
    elif aug_type == 'gaussian':
        _train_transform = v2.Compose([v2.RandomResizedCrop(crop_size),
                                       v2.RandomHorizontalFlip(),
                                       v2.GaussianBlur(kernel_size=(3, 3)),
                                       v2.ToTensor(),
                                       normalize])
    elif aug_type == 'laplacian':
        _train_transform = v2.Compose([v2.RandomResizedCrop(crop_size),
                                       v2.RandomHorizontalFlip(),
                                       v2.RandomAdjustSharpness(sharpness_factor=0.2),
                                       v2.ToTensor(),
                                       normalize])

    train_ds = MyDataset(train_ds, transform=_train_transform)

    return train_ds


def image_test_preprocessing(test_ds: torchvision.datasets.ImageFolder, size, crop_size, normalize):
    _test_transforms = v2.Compose([v2.Resize(size), v2.CenterCrop(crop_size), v2.ToTensor(), normalize])

    test_ds = MyDataset(test_ds, transform=_test_transforms)
    return test_ds


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
