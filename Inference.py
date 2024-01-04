import torch
import torchvision
from transformers import AutoModelForImageClassification, AutoImageProcessor
import numpy as np
import argparse
import data_import
from pathlib import Path
import engine
import evaluate
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--ds_names', default=['MWD', 'ACDC', 'UAVid', 'syndrone'], help='datasets to use during training')
parser.add_argument('--model_path', default="best_models/", type=str, help='The path to the pretrained model')
parser.add_argument('--model_name', default='vit', choices=['vit', 'rn50', 'mob'], type=str, help="model to use in the current run")
parser.add_argument('--batch_size', default=32, type=int, help="batch size")
def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_labels = np.array(['clear', 'fog', 'night', 'rain', 'snow', 'sunrise'])

    # import the checkpoint model and set the path for the pre-trained model (trained using the trainer.py file)
    if args.model_name == 'vit':
        checkpoint = "google/vit-base-patch16-224-in21k"
        folder = Path(args.model_path) / 'ViT/model.pth'
    if args.model_name == 'rn50':
        checkpoint = "microsoft/resnet-50"
        folder = Path(args.model_path) / 'Resnet50/model.pth'
    if args.model_name == 'mob':
        checkpoint = "google/mobilenet_v1_1.0_224"
        folder = Path(args.model_path) / 'mobilenet/model.pth'

    # Load the image processor needed with the information for preprocessing the images for the given models
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    # Import a pre-trained model already with a linear layer attached with an output size equal to the number of labels
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(final_labels),
        ignore_mismatched_sizes=True,
    )

    # Load the model
    model.load_state_dict(torch.load(str(folder)))

    datasets = []

    # Import all the datasets specified in input
    for name in args.ds_names:
        datasets.append(data_import.get_ds(Path(f'./{name}')))

    datasets = np.array(datasets)

    # Get the mapping of the labels
    datasets, ass_ds_final = data_import.map_labels(datasets, final_labels)

    dataloaders = data_import.create_test_dataloader(datasets, image_processor, args.batch_size, ass_ds_final)

    # Set the model to a GPU
    model = model.to(device)

    # Generate the metrics
    clf_metrics = {'precision': evaluate.load("precision"),
                   'recall': evaluate.load("recall"),
                   'f1': evaluate.load("f1"),
                   'accuracy': evaluate.load("accuracy")}

    # Print the results for each dataset
    for name, ds in dataloaders.items():
        print('==========================================================')
        print('Results for dataset', name)
        print(len(ds['ds']))
        res, errors_per_class, conf_mat = engine.test_single_ds(model, ds['ds'], device, clf_metrics, ds['ass'], len(final_labels))
        print(res)
        print(errors_per_class)
        # Print the confusion matrix for each dataset
        conf_mat.plot(cmap=plt.cm.Reds)
        plt.show()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)