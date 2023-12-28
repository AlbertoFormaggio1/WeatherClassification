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

# Get the processor associated with the weights we're going to use
checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
BATCH_SIZE = 16

ds_mwd = data_import.get_ds(Path('./MWD'))
ds_acdc = data_import.get_ds(Path('./ACDC'))


datasets = np.array([ds_mwd, ds_acdc])

final_labels = np.array(['clear', 'fog', 'night', 'rain', 'snow', 'sunrise'])
datasets, ass_ds_final = data_import.map_labels(datasets, image_processor, final_labels, BATCH_SIZE, None)

weights = data_import.compute_weights(datasets)

dataloaders = data_import.gen_dataloader(datasets, image_processor, weights, ass_ds_final, BATCH_SIZE, None)
print(dataloaders)

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
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)

summary(model, depth=6, col_names=['input_size', 'output_size', 'num_params'], input_data=(3, 224, 224))

training_args = TrainingArguments(
    output_dir="./logs/vit-weather",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    report_to='tensorboard',
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataloaders["train"],
    eval_dataset=dataloaders["eval"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate()
# some nice to haves:
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

trainer.push_to_hub()

"""
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.dataset = datasets.ImageFolder(root=self.root)



p = Path('./ACDC')

X = {}
y = {}
ids = {}
for split in os.scandir(p):
    if split.is_dir():
        folder = p / split.name
        X[split.name] = []
        y[split.name] = []
        ids[split.name] = []
        for f in os.scandir(folder):
            # Find all the files ending in .jpg, .png and .jpeg while discarding the others
            if re.match(r'\S+\.(?:jpg|png|jpeg)', f.name):
                extr = str(f.name).split('_')
                # extract the file name
                clas = extr[0]
                id = extr[1].split('.')[0]
                X[split.name].append(plt.imread(Path(f)))
                y[split.name].append(clas)
                ids[split.name].append(id)
                """
