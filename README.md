# maneuver-id

This repository contains the code used to classify U.S. Air Force flight simulator data into "good data" and "bad data" classes using the data presented in Maneuver Identification Challenge presented by Samuel et al. (https://arxiv.org/abs/2108.11503).

# Project Structure 

The project is divided into three directories: analysis, models and utils.

    ./
    ├── /models                     # deep learning models implemented
    │   ├── /FCN.py                     # Fully Convolutional Network
    │   ├── /BCNN.py                    # Bayes Convolutional Network
    │   └── /ResNet.py                  # Time Series ResNet
    ├── /analysis                   # functions that perform statistical tests and performance measures
    |   ├── /data_visualization.ipynb   # notebook to visualize data
    │   ├── /statistics.py              # functions for significance testing
    │   └── /metrics.py                 # functions to calculate performance measures
    └── /utils                      # augmentations, dataloaders, and functions used for training
    │   ├── /augmentations.py           # custom augmentations used for training
    │   ├── /DataLoaders.py             # reads data and delivers dataloaders with a consistent, predetermined split
    │   └── /train_val_funcs.py         # utils for training, validating, and testing
    └── baseline.ipynb             # Implementation of original baseline

## Models

The models directory includes three different models implemented in order to classify the time series data: A Fully Convolutional Network (FCN), Bayes Convolutional Network (BCNN) and a ResNet.

## Analysis

The analysis folder contains the functions used to calculate significance testing and model performance. Everything that relates to the analysis itself is contained in this folder, not the utils.

## Utils

The utils directory contains the utilities needed to train the models. This includes things like the functions used for training, validating, and testing, custom built augmentations used during training, as well as the functions used to load the data.

### Regarding DataLoaders.py

We built the DataLoaders.py as a function that handles loading the Train, Val, and Test data exactly the same way each time it is called. So it handles the process of making a Test data which is held out and if run on different machines it will produce the same held out split every single time.
