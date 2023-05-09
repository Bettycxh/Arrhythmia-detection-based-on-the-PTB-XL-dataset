# DS4Healthcare-Group-4: 
# Arrhythmia classification of 12-lead ECG using deep learning techniques based on the PTB-XL dataset

## Abstract

![img](https://github.com/Bettycxh/SE-MSCNN-A-Lightweight-Multi-scaled-Fusion-Network-for-Sleep-Apnea-Detection-Using-Single-Lead-ECG-/blob/main/pic/model.png)


## Requirements
- Python==3.6
- Keras==2.3.1
- TensorFlow==1.14.0

## Data Preparation
To download the dataset used for training and testing, please refer to [PTB-XL, a large publicly available electrocardiography dataset](https://physionet.org/content/ptb-xl/1.0.1/)

- Download the [ZIP file](https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip)
- Run [example_physionet.py](https://physionet.org/content/ptb-xl/1.0.1/example_physionet.py) to get the data

## Usage
To train the model, execute the python file

- Five classes detection (Normal, MI, STTC, CD, HYP)
  Run [train_5.py](https://github.com/Bettycxh/DS4Healthcare-Group-4/train_5.py)

- Two classes detection (Normal, Arrhythmia)
  Run [train_2.py](https://github.com/Bettycxh/DS4Healthcare-Group-4/train_2.py)

## Email
If you have any questions, please email to: [xhchen@nyu.edu](mailto:xhchen@nyu.edu)
