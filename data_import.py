import torch
from torchvision.transforms import v2
from datasets import load_dataset, DatasetDict
from transformers import AutoImageProcessor
from typing import List, Tuple


def get_ds(ds_name: str, image_processor: AutoImageProcessor, aug_type: str = None) -> DatasetDict:
    ds = load_dataset(ds_name)

    # Splitting the dataset into training and validation.
    # Note that we only have training and test set. If we use the test set for hyperparameter tuning, then we would be
    # overfitting over the test set
    trainval_ds = ds['train'].train_test_split(test_size=0.2)
    prepared_ds = DatasetDict({
        'train': trainval_ds['train'],
        'eval': trainval_ds['test'],
        'test': ds['test']
    })

    image_preprocessing(prepared_ds, image_processor, aug_type)

    return prepared_ds


def image_preprocessing(prepared_ds: DatasetDict, image_processor: AutoImageProcessor, aug_type: str):
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

    image_train_preprocessing(prepared_ds, crop_size, normalize, aug_type)
    image_test_preprocessing(prepared_ds, size, crop_size, normalize)


def image_train_preprocessing(prepared_ds: DatasetDict, crop_size: Tuple[int], normalize: v2.Normalize, aug_type: str):
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

    def train_transforms(examples):
        examples["pixel_values"] = [_train_transform(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

    prepared_ds['train'].set_transform(train_transforms)


def image_test_preprocessing(prepared_ds: DatasetDict, size, crop_size, normalize):
    _test_transforms = v2.Compose([v2.Resize(size), v2.CenterCrop(crop_size), v2.ToTensor(), normalize])
    def test_transforms(examples):
        examples["pixel_values"] = [_test_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

    prepared_ds['eval'].set_transform(test_transforms)
    prepared_ds['test'].set_transform(test_transforms)


def join_ds(datasets: List[DatasetDict]) -> DatasetDict:
    pass