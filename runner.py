import subprocess

rn = '32_base_1_2_0.1_0.0001_pl_50'
vit = '16_base_1_3_0.05_5e-5_pl_30'


model_name = 'rn50'
batch_sizes = [64]
augmentation = ['sharpener']
ds_weight = [2]
class_weight = [1]
label_smoothing = [0.1]
learning_rates = [1e-4]
schedule = ['cosine', 'plateau']
epochs = 50

for b in batch_sizes:
    for a in augmentation:
        for dw in ds_weight:
            for cw in class_weight:
                for l in label_smoothing:
                    for lr in learning_rates:
                        for s in schedule:
                            subprocess.run(f"python trial.py --model={model_name} --batch_size={b} --augmentation={a} "
                                           f"--ds_weight={dw} --class_weight={cw} --label_smoothing={l} "
                                           f"--learning_rate={lr} --scheduler={s} --epochs={epochs}", shell=True)



model_name = 'vit'
batch_sizes = [16]
augmentation = ['base']
ds_weight = [1]
class_weight = [3]
label_smoothing = [0]
learning_rates = [5e-5]
schedule = ['plateau']
epochs = 30

for b in batch_sizes:
    for a in augmentation:
        for dw in ds_weight:
            for cw in class_weight:
                for l in label_smoothing:
                    for lr in learning_rates:
                        for s in schedule:
                            subprocess.run(f"python trial.py --model={model_name} --batch_size={b} --augmentation={a} "
                                           f"--ds_weight={dw} --class_weight={cw} --label_smoothing={l} "
                                           f"--learning_rate={lr} --scheduler={s} --epochs={epochs}", shell=True)


mob = '16_base_1_1_0.1_5e-5_cosdecay_50'

model_name = 'mob'
batch_sizes = [16]
augmentation = ['base']
ds_weight = [3]
class_weight = [3]
label_smoothing = [0.1]
learning_rates = [5e-5]
schedule = ['cosine']
epochs = 20

for b in batch_sizes:
    for a in augmentation:
        for dw in ds_weight:
            for cw in class_weight:
                for l in label_smoothing:
                    for lr in learning_rates:
                        for s in schedule:
                            subprocess.run(f"python trial.py --model={model_name} --batch_size={b} --augmentation={a} "
                                           f"--ds_weight={dw} --class_weight={cw} --label_smoothing={l} "
                                           f"--learning_rate={lr} --scheduler={s} --epochs={epochs}", shell=True)