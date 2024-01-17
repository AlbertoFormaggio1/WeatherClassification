import torch
import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
import cv2
import numpy as np
import os

NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_path: str,
                       test_path: str,
                       batch_size: int,
                       pre_proc_type: str,
                       num_workers: int = NUM_WORKERS):

    # Importing the datasets with imageFolder
    train_ds = HistogramDataset(train_path, pre_proc_type)
    test_ds = HistogramDataset(test_path, pre_proc_type)

    # Creating the dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=False)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, drop_last=False)

    classes = train_ds.classes

    return train_dataloader, test_dataloader, classes


class HistogramDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, preproc_type, loader=default_loader, is_valid_file=None):
        super(HistogramDataset, self).__init__(root=root, loader=loader, is_valid_file=is_valid_file)
        self.pre_proc_type = preproc_type

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        im = cv2.imread(image_path)

        im_nonoise = cv2.GaussianBlur(im, (3, 3), 1)
        if(self.pre_proc_type == 'lab' or self.pre_proc_type=='rgb'):
            if(self.pre_proc_type == 'lab'):
                prep_image = (im_nonoise * 1. / 255).astype(np.float32)
                im_lab = cv2.cvtColor(prep_image, cv2.COLOR_BGR2LAB)
            hist = calc_hists(im_lab, self.pre_proc_type)

            # Setting up a matrix
            hist = np.stack([h for h in hist], axis=-1)
            # hist = np.stack([h for h in hist], axis=-1)
            hist = np.squeeze(hist)

            # Normalizing the vector with L2 normalization
            norm = np.linalg.norm(hist)
            norm_hist = hist / norm
            # you need to convert img from np.array to torch.tensor
            # this has to be done CAREFULLY!
            sample = torchvision.transforms.ToTensor()(norm_hist)
            return sample, target


# Define a function to compute the histogram of the image (channel by channel)
def calc_hists(img: np.ndarray, hist_type) -> list:
    """
    Calculates the histogram of the image (channel by channel).

    Args:
        img (numpy.ndarray): image to calculate the histogram

    Returns:
        list: list of histograms
    """

    assert img.ndim == 3, "The image must have 3 dimensions: (Height,Width,Channels)"

    ch_1 = img[..., 0]
    ch_2 = img[..., 1]
    ch_3 = img[..., 2]

    # Color image
    if hist_type == 'rgb':
        # Get the 3 channels
        # Compute the histogram for each channel. Please, bear in mind that in the "Range" parameter, the upper bound is exclusive. So, for considering values in the range [0,255] we must pass [0,256]. https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
        blue_hist = cv2.calcHist([ch_1], [0], None, [16], [0, 256])
        red_hist = cv2.calcHist([ch_2], [0], None, [16], [0, 256])
        green_hist = cv2.calcHist([ch_3], [0], None, [16], [0, 256])

        return [blue_hist, green_hist, red_hist]
    # Greyscale image
    elif hist_type == 'lab':

        L_hist = cv2.calcHist([ch_1], [0], None, [16], [0, 100])
        a_hist = cv2.calcHist([ch_2], [0], None, [16], [-128, 128])
        b_hist = cv2.calcHist([ch_3], [0], None, [16], [-128, 128])

        return [L_hist, a_hist, b_hist]
    else:
        raise Exception("The image must have either 1 (greyscale image) or 3 (color image) channels")


