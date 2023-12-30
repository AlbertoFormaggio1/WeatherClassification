import os
from datasets import DatasetDict, ClassLabel, Sequence, Value, concatenate_datasets
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import DefaultDataCollator
import evaluate
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from torchsummary import summary
from transformers import AutoImageProcessor
import data_import
import re
from pathlib import Path
import engine
from torch.utils.tensorboard import SummaryWriter
import utils
import argparse
import datetime
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument('--epochs', default=1, type=int, help="Epochs.")
parser.add_argument('--augmentation', default='base', choices=["base", 'geometric_simple', 'geometric_simple_v', 'gaussian', 'laplacian'], type=str, help="Augmentation type to adopt")
parser.add_argument('--model_name', default='mobv2', choices=['vit', 'rn50', 'mobv2'], type=str, help="model to use in the current run")
parser.add_argument('--ds_names', default=['MWD', 'ACDC', 'UAVid', 'syndrone'], type=str, help='datasets to use during training')
parser.add_argument('--learning_rate', default=5e-5, type=float, help="Learning rate.")
parser.add_argument('--ds_weight', default=1, type=float, help='Importance to the data in a minority dataset. Handles dataset imbalance.')
parser.add_argument('--class_weight', default=1, type=float, help='Importance to the data in a minority class. Handles class imbalance.')

def main(args: argparse.Namespace):
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    final_labels = np.array(['clear', 'fog', 'night', 'rain', 'snow', 'sunrise'])

    if args.model_name == 'vit':
        checkpoint = "google/vit-base-patch16-224-in21k"
    if args.model_name == 'rn50':
        checkpoint = "microsoft/resnet-50"
    if args.model_name == 'mobv2':
        checkpoint = "google/mobilenet_v2_1.0_224"

    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(final_labels),
        ignore_mismatched_sizes=True,
    )

    datasets = []

    for name in args.ds_names:
        datasets.append(data_import.get_ds(Path(f'./{name}')))

    datasets = np.array(datasets)


    datasets, ass_ds_final = data_import.map_labels(datasets, final_labels)

    weights = data_import.compute_weights(datasets, args.ds_weight, args.class_weight)

    dataloaders = data_import.gen_dataloader(datasets, image_processor, weights, ass_ds_final, args.batch_size, args.augmentation)

    summary(model, depth=6, col_names=['input_size', 'output_size', 'num_params'], input_data=(3, 224, 224))

    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

    writer = SummaryWriter(log_dir=args.logdir)

    clf_metrics = {'precision': evaluate.load("precision"),
                   'recall': evaluate.load("recall"),
                   'f1': evaluate.load("f1"),
                   'accuracy': evaluate.load("accuracy")}

    results = engine.train(model, dataloaders['train'], dataloaders['eval'], opt, loss_fn, args.epochs, device, clf_metrics, writer)

    for test_ds in dataloaders['test']:
        print('==========================================================')
        print('Results for dataset', test_ds['name'])
        res = engine.test(model, dataloaders['eval'], loss_fn, device, clf_metrics)
        for k, v in res.items():
            writer.add_scalar(f'{k}/test', v)
        print(res)

    writer.flush()
    writer.close()

    torch.save(model, f'{args.logdir}/model.pth')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)