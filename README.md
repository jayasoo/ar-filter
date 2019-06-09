# AR-filter
AR Filters using CNN.

This project builds a convolutional neural network (CNN) to identify facial keypoints of any given face. Once keypoints are identified, AR filters are applied to the face by calculating its positions from the keypoint outputs of the CNN. Using OpenCV, filters are applied to the webcam feed in realtime.

## Dataset
A kaggle dataset is used for building the model. It can be downloaded from here: https://www.kaggle.com/c/facial-keypoints-detection/data.

## Usage

Download the dataset and add it in the project directory.

Training: ```python train.py```

Apply filter to the webcam feed by running ```python filter_cam.py```
