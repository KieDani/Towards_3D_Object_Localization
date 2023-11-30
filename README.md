# Towards Learning Monocular 3D Object Localization From 2D Labels using the Physical Laws of Motion

This repository is the official implementation of the paper "Towards Learning Monocular 3D Object Localization From 2D Labels using the Physical Laws of Motion" (see [here](https://arxiv.org/abs/2310.17462)). We are happy to announce that our paper has been accepted at the International Conference on 3D Vision 2024 (3DV 2024).

## Preparation
We provide the datasets and the [model weights](https://mediastore.rz.uni-augsburg.de/get/h8bXv437nS/) to reproduce the paper results. The datasets can be downloaded using the bash script download_datasets.sh. 
After downloading the datasets and models, you have to extract the file dataset.zip to <path_to_datasets> and the file checkpoints.zip to <path_to_models> respectively.
Then, you have to set the following variables in paths.py:
```paths
data_path = <path_to_datasets>
checkpoint_path = <path_to_models>
logs_path = <path_to_logs>
```
In <path_to_logs> the logs, checkpoints and evaluation metrics of your runs will be saved.

## Requirements
First, install the current version of pytorch with:
```setup
pip install torch torchvision
```
Now, you can install the other requirements with:
```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python -m general.experiments --env RD
python -m general.experiments --env SD-S
python -m general.experiments --env SD-M
python -m general.experiments --env SD-L
```
In case you want to run the experiments with other settings, check out 
```train
python -m general.experiments --help
```

## Evaluation

To reproduce the paper results, run:
```eval
python -m general.evaluate
```

To evaluate your own trained models, run:
```eval
python -m general.evaluate --path <path_to_model>
```
<path_to_model> is the full path to the checkpoints. An example for training and evaluating a model is:
```eval
python -m general.experiments --env RD`
python -m general.evaluate --path <logs_path>/checkpoints/checkpoints/review/realball/sintitle:resnet34_lossmode:2Dpred_nograd_<date&time>
```
<date&time> is the date and time you started the training. You need to look up the path in the logs folder.
The results are printed in the console and additionally saved in <logs_path>/eval. The average metrics per environment is denoted as "dtgs_env" and the average metrics per camera location per environment is denoted as "dtgs_cam". 


