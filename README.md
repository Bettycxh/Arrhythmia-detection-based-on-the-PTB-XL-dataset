# DS4Healthcare-Group-4: ECG-based arrhythmia detection with deep learning network

<!-- ## Abstract
Sleep apnea (SA) is a common sleep disorder that occurs during sleep, leading to the decrease of oxygen saturation in the blood. It would develop a variety of complications like diabetes, chronic kidney disease, depression, cardiovascular diseases, or even sudden death. Early SA detection can help physicians to take interventions for SA patients to prevent malignant events. This paper proposes a lightweight SA detection method of multi-scaled fusion network named SE-MSCNN based on single-lead ECG signals. The proposed SE-MSCNN mainly includes multi-scaled convolutional neural network (CNN) module and channel-wise attention module. In order to facilitate the SA detection performance, various scaled ECG information with different-length adjacent segments are extracted by three sub-neural networks. To overcome the problem of local concentration of feature fusion with concatenation, a channel-wise attention module with a squeeze-to-excitation block is employed to fuse the different scaled features adaptively. Furthermore, the ablation study and computational complexity analysis of the SE-MSCNN are conducted. Extensive experiment results show that the proposed SE-MSCNN has the performance superiority to the state-of-the-art methods for SA detection on the Apnea-ECG benchmark dataset. The SE-MSCNN with the merits of quick response and lightweight parameters can be potentially embedded into a wearable device to provide an SA detection service for individuals in home sleep test (HST).
![img](https://github.com/Bettycxh/SE-MSCNN-A-Lightweight-Multi-scaled-Fusion-Network-for-Sleep-Apnea-Detection-Using-Single-Lead-ECG-/blob/main/pic/model.png)
 -->

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

- Five classes detection
  Run [train_5.py]
<!--(https://github.com/Bettycxh/) -->

- Two classes detection
  Run [train_2.py]
<!--(https://github.com/Bettycxh/) -->

## Email
If you have any questions, please email to: [xhchen@nyu.edu](mailto:xhchen@nyu.edu)
