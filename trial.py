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

ds_mwd = data_import.get_ds('MWD', image_processor, None)
print(ds_mwd)
ds_acdc = data_import.get_ds('ACDC', image_processor, None)
print(ds_acdc)

labels_mwd = ds_mwd['train'].features['label'].names
labels_acdc = ds_acdc['train'].features['label'].names
label2id_mwd = {c: i for i, c in enumerate(labels_mwd)}
label2id_acdc = {c: i for i, c in enumerate(labels_acdc)}

labels = np.array(['clear', 'fog', 'night', 'rain', 'snow', 'sunrise'])
label_ass_acdc = {}
for l in labels_acdc:
    ind = np.where(labels == l)
    label_ass_acdc[str(label2id_acdc[l])] = str(ind[0][0])

label_ass_mwd = {}
ass_lab_mwd = {'shine': 'clear', 'cloudy': 'fog', 'rain': 'rain', 'sunrise': 'sunrise'}
for l in labels_mwd:
    true_l = ass_lab_mwd[l]
    ind = np.where(labels == true_l)
    label_ass_mwd[str(label2id_mwd[l])] = str(ind[0][0])

features = ds_acdc['train'].features.copy()
del features['label']
features['label'] = ClassLabel(names=labels.tolist())
features['pixel_values'] = Sequence(feature=Sequence(feature=Sequence(feature=Value(dtype='float32', id=None), length=-1, id=None), length=-1, id=None), length=-1, id=None)
def change_label(x, label_ass):
    x['label'] = int(label_ass[str(x['label'])])
    return x

ds_mwd = ds_mwd.map(change_label, fn_kwargs={'label_ass': label_ass_mwd}, num_proc=16, features=features)
ds_mwd = ds_mwd.remove_columns('image')
ds_mwd['train'] = ds_mwd['train'].add_column('ds_name', ['mwd'] * len(ds_mwd['train']))
ds_mwd['eval'] = ds_mwd['eval'].add_column('ds_name', ['mwd'] * len(ds_mwd['eval']))

ds_acdc = ds_acdc.map(change_label, fn_kwargs={'label_ass': label_ass_acdc}, num_proc=16, features=features)
ds_acdc = ds_acdc.remove_columns('image')
ds_acdc['train'] = ds_acdc['train'].add_column('ds_name', ['acdc'] * len(ds_acdc['train']))
ds_acdc['eval'] = ds_acdc['eval'].add_column('ds_name', ['acdc'] * len(ds_acdc['eval']))


print(ds_acdc['train'].features)

prepared_ds = DatasetDict(
    {'train': concatenate_datasets([ds_mwd['train'], ds_acdc['train']]),
     'eval': concatenate_datasets([ds_mwd['eval'], ds_acdc['eval']]),
     }
)

prepared_ds['train'] = prepared_ds['train'].shuffle()

print(prepared_ds['train'])
data_collator = DefaultDataCollator()
# import the metric we're going to use
accuracy = evaluate.load("accuracy")

# Create a function to compute the metric we decided to use
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


labels = ds_mwd['train'].features['label'].names

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
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["eval"],
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