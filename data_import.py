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
from sklearn.model_selection import train_test_split

NUM_WORKERS = os.cpu_count()


def map_labels(datasets: List[dict], final_labels: List[str]):
    """
    Maps the label from the original datasets to the label used for training the model, associating the similar labels with one another
    :param datasets: the datasets used for training the model
    :param final_labels: the labels used for the training
    :return: tuple (re-mapped datasets, inverse assignment)
    """
    labels_ds = []
    label2id_ds = []
    label_ass_ds = []
    ass_ds_final = []
    inverse_ass = []
    for ds in datasets:
        labels_ds.append(ds['classes'])     # Save the labels of the current dataset
        label2id_ds.append({c: i for i, c in enumerate(labels_ds[-1])})     # Save the mapping label to id (i.e. "sunrise" -> 3)

        # Append the mapping from the dataset labeling to the "generalized" labeling used for the model
        if 'shine' in labels_ds[-1]:
            ass_ds_final.append({'shine': 'clear', 'cloudy': 'fog', 'rain': 'rain', 'sunrise': 'sunrise'})  # MWD ds
        elif 'snow' in labels_ds[-1]:
            ass_ds_final.append(
                {'clear': 'clear', 'fog': 'fog', 'rain': 'rain', 'night': 'night', 'snow': 'snow'})  # ACDC ds
        else:
            ass_ds_final.append({'clear': 'clear', 'fog': 'fog', 'rain': 'rain', 'night': 'night'}) # Uav and syndrone

    # Generate the mapping that goes from the DATASET LABEL ID -> GENERALIZED LABEL ID
    # In this case, the generalized label id is the position of the label name inside the array of the generalized labels.
    # At the end, for each dataset, you will have a dictionary. For example: {0:1, 1:2, 2:0}, which means that the label 0
    # of the original dataset needs to be mapped to the label 1 for training the NN
    for idx, l_ds in enumerate(labels_ds):
        label_ass_ds.append({})
        for label in l_ds:
            # Get, for the current dataset (idx), the label name in the generalized setting
            true_l = ass_ds_final[idx][label]
            # Get the position of the general label in the array of labels (the label ID)
            ind = np.where(true_l == final_labels)
            label_ass_ds[-1][label2id_ds[idx][label]] = ind[0][0]

        # Save also the inverse assignment in order to have a mapping for when doing inference
        inverse_ass.append({v: k for k, v in label_ass_ds[-1].items()})

    # replace the old label with the new one
    for idx, ds in enumerate(datasets):
        # replace targets over train and test set
        for key, split in ds.items():
            if key == 'train' or key == 'test':
                # Get the targets
                targets = np.array(split.targets)
                # Do a copy of the targets to avoid that previous substitutions influence future ones
                new_targets = targets.copy()
                # For each pair (old label, new label) replace all the targets old_label with new_label
                for (old_l, new_l) in label_ass_ds[idx].items():
                    new_targets[targets == old_l] = new_l
                split.targets = new_targets.tolist()

    return datasets, inverse_ass


def compute_weights(datasets, alpha_ds, alpha_class):
    """
    Computes the weights for each sample according to the formula:
    alpha_ds * ds_weight + alpha_class * class_weight.
    Where alpha_ds and alpha_class give the tradeoff of importance between class imbalance and dataset imbalance
    :param datasets: the datasets containing the samples
    :param alpha_ds: weight for the dataset
    :param alpha_class: weight for the class
    :return: a weight for each sample
    """
    # The weight for the dataset imbalance is computed as 1 / |DATASET|, giving thus more importance to small datasets
    len_train = [1 / float(len(ds['train'])) for ds in datasets]
    weights = alpha_ds * torch.tensor(len_train, dtype=torch.float32)

    # Assign to each sample a weight by repeating it according to the length of each dataset
    weights_ds = []
    for ds, w in zip(datasets, weights):
        weights_ds.append(w.repeat(len(ds['train'])))

    targ = [ds['train'].targets for ds in datasets]     # Get the targets for each class
    # Concatenate the targets of each dataset (you want to compute the weights based on the merged dataset distribution)
    # (NOTE: we don't merge before the datasets because after getting a dataloader we can't do any modification)
    targets = torch.cat([torch.tensor(t) for t in targ])
    classes = torch.unique(targets)

    # Compute the weights w(c) = 1 / |Num samples of class c|
    class_sum = torch.tensor([torch.sum(targets == c) for c in classes])
    inv_class_sum = 1 / class_sum
    for t, wds in zip(targ, weights_ds):
        t = torch.tensor(t)
        for c, wc in zip(classes, inv_class_sum):
            wds[(t == c).nonzero()] += alpha_class * wc

    # Concatenate the weights for each dataset in a unique tensor
    full_w = torch.tensor([])
    for w in weights_ds:
        full_w = torch.cat([full_w, w])

    return full_w


def gen_dataloader(datasets, image_processor, weights, ass_ds_final, batch_size, aug_type):
    """
    Generates the dataloaders for each split (train, eval, test) for each dataset
    :param datasets: the datasets splitted appropriately
    :param image_processor: image processor with the info on how to process each image
    :param weights: weights of the samples
    :param ass_ds_final: the mapping to save in the final dictionary (useful for re-mapping the labels during inference)
    :param batch_size: batch size
    :param aug_type: augmentation type used for training ONLY
    :return: a dictionary {'train': train_dl, 'eval': eval_dl, 'test': [{'ds': test_dl[i], 'ass': ass_ds_final[i] for i in len(datasets)]}
    """
    dataloaders = {}

    # Merge the datasets
    ds_to_merge = [ds['train'] for ds in datasets]
    merged_train_ds = torch.utils.data.ConcatDataset(ds_to_merge)

    """
    ds = [ds for ds in merged_train_ds.datasets]
    all_labels = np.concatenate([p for p in [d.targets for d in ds]])
    for idx, (pattern, y) in enumerate(zip(merged_train_ds, all_labels)):
        merged_train_ds[idx] = (pattern[0], y)
    print(next(iter(merged_train_ds)))
    """

    # Define a generator (setting the random seed for reproducibility of the results
    generator = torch.Generator().manual_seed(42)
    # Split in the same way samples and weights (important that the manual seed is the same, otherwise the weights
    # wouldn't be mapped to the correct samples anymore)
    indices_train, indices_eval = train_test_split(range(len(merged_train_ds)), test_size=0.2, train_size=0.8, random_state=42)

    # Preprocessing of training and test data
    pre_train_ds = image_preprocessing(merged_train_ds, image_processor, aug_type, 'train', indices_train)
    pre_eval_ds = image_preprocessing(merged_train_ds, image_processor, aug_type, 'eval', indices_eval)

    # Generate a weighted sampler oversampling those samples belonging to the minority class
    sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights=weights[indices_train], num_samples=len(indices_train), replacement=True, generator=generator)

    # Create the dataloaders
    train_dataloader = DataLoader(pre_train_ds, sampler=sampler_train, batch_size=batch_size, num_workers=NUM_WORKERS,
                                  drop_last=False)
    eval_dataloader = DataLoader(pre_eval_ds, batch_size=batch_size, num_workers=NUM_WORKERS,
                                 drop_last=False)

    dataloaders['train'] = train_dataloader
    dataloaders['eval'] = eval_dataloader
    dataloaders['test'] = {}

    # Do the same for each test dataset separately (we are interested in the accuracy of our model over each dataset)

    dataloaders['test'] = create_test_dataloader(datasets, image_processor, batch_size, ass_ds_final)

    return dataloaders

def create_test_dataloader(datasets, image_processor, batch_size, ass_ds_final):
    dataloaders = {}
    for idx, ds in enumerate(datasets):
        pre_test_ds = image_preprocessing(ds['test'], image_processor, None, 'test', None)

        test_dataloader = DataLoader(pre_test_ds, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False,
                                     drop_last=False)

        dataloaders[str(ds['name'])] = {'ds': test_dataloader, 'ass': ass_ds_final[idx]}

    return dataloaders

def get_ds(ds_name: Path):
    """
    Loads the dataset at the path specified in input
    :param ds_name: the path in memory to the dataset
    :return: {"train": train_ds, "test": test_ds, "classes": class_names, 'name': ds_name}
    """
    train_ds = tv.datasets.ImageFolder(str(ds_name / 'train'))
    test_ds = tv.datasets.ImageFolder(str(ds_name / 'test'))

    classes = train_ds.classes

    return {"train": train_ds, "test": test_ds, "classes": classes, 'name': ds_name}


def image_preprocessing(ds, image_processor: AutoImageProcessor, aug_type: str, type: str, indices):
    """
    Performs the preprocessing of the given dataset
    :param ds: dataset to preprocess (only the split)
    :param image_processor: the processor to use
    :param aug_type: augmentation type
    :param type: train or eval/test. Training augmentation must be difference from the one used for evaluation
    :return: preprocessed ds split
    """
    # Normalize the features with the mean and std dev used
    normalize = v2.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

    # Get the size for the resizing and cropping
    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
        max_size = None
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)
        max_size = image_processor.size.get("longest_edge")

    if type == 'train':
        ds_proc = image_train_preprocessing(ds, crop_size, normalize, aug_type, indices)
    elif type == 'eval':
        ds_proc = image_eval_preprocessing(ds, size, crop_size, normalize, indices)
    else:
        ds_proc = image_test_preprocessing(ds, size, crop_size, normalize)

    return ds_proc


def image_train_preprocessing(train_ds: torchvision.datasets.ImageFolder, crop_size: Tuple[int],
                              normalize: v2.Normalize, aug_type: str, indices):
    # define the transform based on the desired augmentation method
    if aug_type == 'base':
        _train_transform = v2.Compose([v2.RandomResizedCrop(crop_size),
                                       v2.RandomHorizontalFlip(),
                                       v2.ToTensor(),
                                       normalize])
    elif aug_type == 'geometric_simple':
        _train_transform = v2.Compose([v2.RandomResizedCrop(crop_size),
                                       v2.RandomHorizontalFlip(),
                                       v2.RandomRotation(degrees=20),
                                       lambda x: v2.functional.adjust_brightness(x, np.random.uniform(0.75, 1.25, 1)),
                                       v2.ToTensor(),
                                       normalize
                                       ])
    elif aug_type == 'geometric_simple_v':
        _train_transform = v2.Compose([v2.RandomResizedCrop(crop_size),
                                       v2.RandomHorizontalFlip(),
                                       v2.RandomVerticalFlip(p=0.2),
                                       v2.RandomRotation(degrees=20),
                                       lambda x: v2.functional.adjust_brightness(x, np.random.uniform(0.75, 1.25, 1)),
                                       v2.ToTensor(),
                                       normalize
                                       ])
    elif aug_type == 'gaussian':
        _train_transform = v2.Compose([v2.RandomResizedCrop(crop_size),
                                       v2.RandomHorizontalFlip(),
                                       v2.GaussianBlur(kernel_size=(3, 3)),
                                       v2.ToTensor(),
                                       normalize])
    elif aug_type == 'sharpener':
        _train_transform = v2.Compose([v2.RandomResizedCrop(crop_size),
                                       v2.RandomHorizontalFlip(),
                                       v2.RandomAdjustSharpness(sharpness_factor=np.random.uniform(1,3, 1)),
                                       v2.ToTensor(),
                                       normalize])
    elif aug_type == 'none':
        _train_transform = v2.Compose([v2.Resize(crop_size),
                                       v2.ToTensor(),
                                       normalize])


    targets = np.concatenate([x.targets for x in train_ds.datasets])
    train_ds = MyDataset(train_ds, indices, targets[indices], transform=_train_transform)

    return train_ds


def image_eval_preprocessing(eval_ds: torchvision.datasets.ImageFolder, size, crop_size, normalize, indices):
    _eval_transforms = v2.Compose([v2.Resize(size), v2.CenterCrop(crop_size), v2.ToTensor(), normalize])

    targets = np.concatenate([x.targets for x in eval_ds.datasets])
    new_eval_ds = MyDataset(eval_ds, indices, targets[indices], transform=_eval_transforms)

    return new_eval_ds


def image_test_preprocessing(test_ds: torchvision.datasets.ImageFolder, size, crop_size, normalize):
    _test_transforms = v2.Compose([v2.Resize(size), v2.CenterCrop(crop_size), v2.ToTensor(), normalize])

    test_ds = MyDataset(test_ds, None, test_ds.targets, transform=_test_transforms)
    return test_ds


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, ds, indices, labels, transform=None):
        if indices is not None:
            self.subset = torch.utils.data.Subset(ds, indices)
        else:
            self.subset = ds
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        x = self.subset[index][0]
        y = self.labels[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
