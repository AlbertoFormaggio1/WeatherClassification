# Weather Classification

### Abstract
Classifying weather images is a relevant task when considering applications like autonomous driving, in which it can be used for understanding a level of caution to apply or intelligent drone autonomy.
Relying on humans for this task can be inconsitent and lead to inaccurate results and relying on the weather forecast is not reliable as well and can't provide results in real time.
In this work, we exploited the power of Transfer Learning applied to Convolutional Neural Networks (CNNs) and a Visual Transformer to infer a model that is robust and can be applied in multiple applications: from images taken from the windshield of a car to aerial images shot by flying drones.
Different techniques for data augmentation were used to test whether it were possible to increase the robustness of the models by modifying the input data.
We also trained a linear model to see if it were possible to get a better generalization over images belonging to different datasets.

### Instructions to run this repository

1. Clone the repository
```
git clone https://github.com/AlbertoFormaggio1/WeatherClassification.git
```
2. Download the datasets from the link reported in the assignment pdf
3. Download the folder at the following link containing the best models if you want to test them: https://drive.google.com/file/d/17mnNzI9UR8P-wMP6tZV8Fi-Ijp_KiGmP/view?usp=sharing and extract it.
4. For each dataset that doesn't have a folder for each class inside ```train``` and ```test``` run the following (from inside the train/test folder):
```
mkdir clear
mkdir night
mkdir fog
mkdir rain
mkdir snow # (if ACDC)

# For ACDC
mv clear_* clear
mv rain_* rain
mv night_* night
mv fog_* fog
mv snow_* snow

# For UAVid
mv day* clear
mv rain_* rain
mv night_* night
mv fog_* fog

# For Syndrone
mv *ClearNoon* clear
mv *Rain* rain
mv *Night* night
mv *Foggy* fog
```

At the end the folder structure must be something like:
```
workspace
  |_files.py
  |_MWD
  |    |_train
  |         |_shine
  |         |_cloudy
  |         |_rain
  |         |_sunrise
  |    |_test
  |         |_shine
  |         |_cloudy
  |         |_rain
  |         |_sunrise
  |_ACDC
  |    |_train
  |         |_clear
  |         |_night
  |         |_rain
  |         |_fog
  |         |_snow
  |    |_test
  |         |_clear
  |         |_night
  |         |_rain
  |         |_fog
  |         |_snow
  .
  .
  .
  |_logs
  |     |_logfolder1
  |     |_logfolder2
  .
  .
  |_best_models
      |_mobilenet
           |_model.pth
      |_ResNet50
           |_model.pth
      |_ViT
           |_model.pth
```

### The files of this repository
- If you want to train a single model, open ```trainer.py``` and see the parameters that are available at the top of the file.
You can modify the default value inside the .py file or, when running the file, you can run 
```
python trainer.py --param_name1=param_value1 ----param_name2=param_value2 ...
```
You can see an example of this in the ```runner.py```

- If you want to train multiple models with a given set of parameters, you can set the parameters from the source file and then run
```
python runner.py
```

- If you want to run inference and test a model you previously trained, you can run
```
python inference.py --model_path=path_to_the_folder_where_the_file_pth_is_stored  --model_name=model_name
```

There are also 2, non executable files:
- engine.py: contains the methods for running training and inference.
- data_import.py: contains the methods for appropriately importing the data and create a single, merged, dataset

### For the tutor/professor

This part was done mainly in PyTorch to acquire more expertise with the framework as it is very helpful and required in real-life.
I also wanted to test myself a bit on something that was strictly related to OpenCV by using the histograms:
You can launch trainer_hist.py to train the MLP defined in model_mlp.py. The histograms are generated inside an ad-hoc created dataset of pytorch inside the file load_ds_histogram.py

I would have explored this histogram part further but having 4 projects it was tough since the first part on the CNNs took me almost 2 weeks. I tried to start with HuggingFace but its flexibility was too low and I lost much
time on that before switching to vanilla pytorch and trying to understand well how it works.

### For seeing the results of my trainings:
I used tensorboard to keep track of the results of my runs and finding out the best model.
After installing tensorboard, from the folder of this repository run
```
tensorboard --logdir logs
```

Now, by opening a browser and digiting *localhost:6006* in the address bar you can see my results under the "Scalars" section (you can find it at the top left of the screen).

*Note*: the logs of all the runs are not reported in their entirety. Some were removed during the experimenting before thinking about adding them to the GitHub repository, re-training the networks from scratch would have been too expensive. Nevertheless, many of them are still present.

## Important Note
If you want to change the datasets used for training the models, please remember to change the list at the beginning of trainer.py and inference.py with the labels used by the datasets.


