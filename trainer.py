import os
import torch
import numpy as np
import evaluate
from transformers import AutoModelForImageClassification, AutoImageProcessor
from torchsummary import summary
import data_import
import re
from pathlib import Path
import engine
from torch.utils.tensorboard import SummaryWriter
import argparse
import datetime
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument('--epochs', default=30, type=int, help="Epochs.")
parser.add_argument('--augmentation', default='base', choices=["base", 'geometric_simple', 'geometric_simple_v', 'gaussian', 'sharpener'], type=str, help="Augmentation type to adopt")
parser.add_argument('--model_name', default='mob', choices=['vit', 'rn50', 'mob'], type=str, help="model to use in the current run")
parser.add_argument('--ds_names', default=['MWD', 'ACDC', 'UAVid', 'syndrone'], type=str, help='datasets to use during training')
parser.add_argument('--learning_rate', default=5e-5, type=float, help="Learning rate.")
parser.add_argument('--ds_weight', default=1, type=float, help='Importance to the data in a minority dataset. Handles dataset imbalance.')
parser.add_argument('--class_weight', default=1, type=float, help='Importance to the data in a minority class. Handles class imbalance.')
parser.add_argument('--label_smoothing', default=0.1, type=float, help='Label smoothing.')
parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosine', 'plateau'], help='The type of scheduler to use for decreasing the learning rate')


def main(args: argparse.Namespace):
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Initialize the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Define the set of labels of the merged datasets
    # !! MUST BE DEFINED MANUALLY WHEN CHANGING THE DATASET !!
    final_labels = np.array(['clear', 'fog', 'night', 'rain', 'snow', 'sunrise'])

    # Initialize the weights according to the chosen model
    if args.model_name == 'vit':
        checkpoint = "google/vit-base-patch16-224-in21k"
    if args.model_name == 'rn50':
        checkpoint = "microsoft/resnet-50"
    if args.model_name == 'mob':
        checkpoint = "google/mobilenet_v1_1.0_224"

    # Load the image processor needed with the information for preprocessing the images for the given models
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    # Import a pre-trained model already with a linear layer attached with an output size equal to the number of labels
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(final_labels),
        ignore_mismatched_sizes=True,
    )

    # ------------------- DATASET IMPORT -------------------

    datasets = []

    # Import all the datasets specified in input
    for name in args.ds_names:
        datasets.append(data_import.get_ds(Path(f'./{name}')))

    datasets = np.array(datasets)

    # Map the labels as described in the report: this way labels associated to different classes but similar are merged and thus trained together
    datasets, ass_ds_final = data_import.map_labels(datasets, final_labels)

    # Compute the weights for each sample according to the size of the dataset and the size of the overall size of the class
    weights = data_import.compute_weights(datasets, args.ds_weight, args.class_weight)

    # Generate the weighted dataloaders and perform the augmentation
    dataloaders = data_import.gen_dataloader(datasets, image_processor, weights, ass_ds_final, args.batch_size, args.augmentation)

    #summary(model, depth=6, col_names=['input_size', 'output_size', 'num_params'], input_data=(3, 224, 224))

    # Set the model to a GPU
    model = model.to(device)


    # We use cross entropy loss since this is multi-class classification
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    if args.scheduler == 'plateau':
        lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.1, patience=5, threshold=0.08, verbose=True)
    elif args.scheduler == 'cosine':
        steps = args.epochs * (np.ceil(len(dataloaders['train']) / float(args.batch_size)))
        lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-7)
    else:
        lr_schedule = None

    # Generate the writer to print the results of our classification to tensorboard
    writer = SummaryWriter(log_dir=args.logdir)

    # Generate the metrics
    clf_metrics = {'precision': evaluate.load("precision"),
                   'recall': evaluate.load("recall"),
                   'f1': evaluate.load("f1"),
                   'accuracy': evaluate.load("accuracy")}

    # Train the model
    results = engine.train(model, dataloaders['train'], dataloaders['eval'], opt, loss_fn, args.epochs, device, clf_metrics, writer, args.model_name, args.logdir, lr_schedule)

    # To do inference you can run the following line:
    # results = engine.do_inference(model, dataloaders['test']['MWD']['ds'], device, dataloaders['test']['MWD']['ass'])

    # For each dataset print the results
    for name, ds in dataloaders['test'].items():
        print('==========================================================')
        print('Results for dataset', name)
        print(ds)
        print(len(ds['ds']))
        res, _, _ = engine.test_single_ds(model, ds['ds'], device, clf_metrics, ds['ass'], len(final_labels))
        for k, v in res.items():
            writer.add_scalar(f'{args.model_name}/{name}/{k}/test', v, 1)
        print(res)

    # Write all the pending scalars in the writer and then close the writer
    writer.flush()
    writer.close()

    # Save the model
    torch.save(model.state_dict(), f'{args.logdir}/model.pth')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)


