import torch, torchvision
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt
import evaluate
from torch.utils.tensorboard import SummaryWriter
import engine
import load_ds_histogram, model_mlp
import os
import re
from pathlib import Path
import warnings
import argparse
import datetime

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument('--epochs', default=30, type=int, help="Epochs.")
parser.add_argument('--learning_rate', default=0.1, type=float, help="Learning rate.")
parser.add_argument('--label_smoothing', default=0.1, type=float, help='Label smoothing.')
parser.add_argument('--preproc_type', default='lab', type=str, choices=['lab', 'rgb'], help='Type of preprocessing')
parser.add_argument('--hidden_size', default=32, type=int, help='Number of hidden neurons in the MLP')
parser.add_argument('--only_inference', default=True, type=bool, help='Number of hidden neurons in the MLP')

def main(args: argparse.Namespace):
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_path = Path('syndrone')
    train_path = data_path / 'train'
    test_path = data_path / 'test'

    train_dataloader, test_dataloader, classes = load_ds_histogram.create_dataloaders(train_path, test_path, args.batch_size, args.preproc_type)

    mlp = model_mlp.MLP(16 * 3, args.hidden_size, len(classes))

    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(mlp.parameters(), lr=args.learning_rate)

    writer = SummaryWriter(log_dir=args.logdir)

    # Generate the metrics
    clf_metrics = {'precision': evaluate.load("precision"),
                   'recall': evaluate.load("recall"),
                   'f1': evaluate.load("f1"),
                   'accuracy': evaluate.load("accuracy")}


    if (not args.only_inference):
        engine.train(mlp, train_dataloader, test_dataloader, optimizer, loss_fn, args.epochs, device, clf_metrics, 0, writer, 'mlp',args.logdir)

    mlp.load_state_dict(torch.load(str(f'{args.logdir}/model.pth')))

    # Now test on UAVid
    data_path = Path('UAVid')
    train_path = data_path / 'train'
    test_path = data_path / 'test'

    train_dataloader, test_dataloader, classes = load_ds_histogram.create_dataloaders(train_path, test_path, args.batch_size, args.preproc_type)

    results = engine.test_step(mlp, test_dataloader, loss_fn, device, clf_metrics)

    for k, v in results.items():
        writer.add_scalar(f'test/{k}', v, 1)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)