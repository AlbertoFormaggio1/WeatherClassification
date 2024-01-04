# Weather Classification

### Abstract
Classifying weather images is a relevant task when considering applications like autonomous driving, in which it can be used for understanding a level of caution to apply or intelligent drone autonomy.
Relying on humans for this task can be inconsitent and lead to inaccurate results and relying on the weather forecast is not reliable as well and can't provide results in real time.
In this work, we exploited the power of Transfer Learning applied to Convolutional Neural Networks (CNNs) and a Visual Transformer to infer a model that is robust and can be applied in multiple applications: from images taken from the windshield of a car to aerial images shot by flying drones.
Different techniques for data augmentation were used to test whether it were possible to increase the robustness of the models by modifying the input data.

### Instructions to run this repository

1. Clone the repository
```
git clone https://github.com/AlbertoFormaggio1/WeatherClassification.git
```
2. Download the datasets from the link reported in the assignment pdf
3. For each dataset that doesn't have a folder for each class inside ```train``` and ```test``` run the following (from inside the train folder):
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
mv *Noon* clear
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

### For seeing the results of my trainings:
I used tensorboard to keep track of the results of my runs and finding out the best model.
After installing tensorboard, from the folder of this repository run
```
tensorboard --logdir logs
```

Now, by opening a browser and digiting *localhost:6006* in the address bar you can see my results under the "Scalars" section (you can find it at the top left of the screen).


## Important Note
If you want to change the datasets used for training the models, please remember to change the list at the beginning of trainer.py and inference.py with the labels used by the datasets.


