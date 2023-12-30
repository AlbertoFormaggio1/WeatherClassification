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


dataset_names = ['MWD', 'ACDC', 'UAVid', 'syndrone']
model_name = 'vit'
augmentation = 'base'



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Get the processor associated with the weights we're going to use
checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
BATCH_SIZE = 16

datasets = []

for name in dataset_names:
    datasets.append(data_import.get_ds(Path(f'./{name}')))

datasets = np.array(datasets)

final_labels = np.array(['clear', 'fog', 'night', 'rain', 'snow', 'sunrise'])
datasets, ass_ds_final = data_import.map_labels(datasets, image_processor, final_labels, BATCH_SIZE, augmentation)

weights = data_import.compute_weights(datasets)

dataloaders = data_import.gen_dataloader(datasets, image_processor, weights, ass_ds_final, BATCH_SIZE, augmentation)

for batch in dataloaders['train']:
    print(batch)
    break

data_collator = DefaultDataCollator()
# import the metric we're going to use
accuracy = evaluate.load("accuracy")


# Create a function to compute the metric we decided to use
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


labels = final_labels

model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
)

summary(model, depth=6, col_names=['input_size', 'output_size', 'num_params'], input_data=(3, 224, 224))

model = model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
EPOCHS = 1

writer = SummaryWriter(log_dir='logs/vit-weather')

clf_metrics = {'precision': evaluate.load("precision"),
               'recall': evaluate.load("recall"),
               'f1': evaluate.load("f1"),
               'accuracy': evaluate.load("accuracy")}

results = engine.train(model, dataloaders['train'], dataloaders['eval'], opt, loss_fn, EPOCHS, device, clf_metrics, writer)

for test_ds in dataloaders['test']:
    print('==========================================================')
    print('Results for dataset', test_ds['name'])
    res = engine.test(model, dataloaders['eval'], loss_fn, device, clf_metrics)
    for k,v in res.items():
        writer.add_scalar(f'{k}/test', v)
    print(res)

writer.flush()
writer.close()