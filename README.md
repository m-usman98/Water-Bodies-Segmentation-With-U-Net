# Water-Bodies-Segmentation-With-U-Net
Accurate segmentation of water bodies from satellite imagery is vital for environmental monitoring and resource management. This study utilizes the "Satellite Images of Water Bodies" dataset from Kaggle, which features labeled satellite images for this purpose. We apply the U-Net architecture, known for its effectiveness in image segmentation tasks, to identify and delineate water bodies within these images. The dataset is divided into 80% for training and 20% for testing, allowing for a comprehensive evaluation of the model's performance in accurately detecting and segmenting water features.

| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp;  GT &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; GT Mask&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  |  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  Prediction &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;|
|--------|--------|--------|

<p align="center">
  <img src="https://github.com/m-usman98/Water-Bodies-Segmentation-With-U-Net/blob/main/Output/2.jpg" width="1200"/>
</p>

## Dataset
The dataset can be downloaded from [here](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies) To prepare the dataset, ensure that the training set consists of 80% of the images, including both the ground truth (GT) and mask images, while the remaining 20% will be used for testing.

## Libraries Requirement
The following libraries are required to train your model.

```angular2html
torch 1.13
visdom 0.2.3
```

## Training
You can start training by executing the following command.
  ```python
python train.py
```
