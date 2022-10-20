# Cat-breed-images-classification
## README-File
-
-
-

## Executive Summary:


## Table of Contents:
1. [Introduction](#introduction)
2. [Assumption](#assumption)
3. [Data Set](#dataset)
4. [Data Augmentation](#augment)
5. [CNN Backbones](#bb)
7. [Network Architecture of each CNN](#architecture)
8. [Training and Fine Tuning](#finetuning)
9. [Discussion](#discussion)
10. [Conclusion](#conclusion)
11. [Reference](#reference)

## Introduction: <a name="introduction"></a>
- The purpose of this project is to train an image classification (single-label) with our custom dataset (Cat breed images) with CNN models pretrained.
- There are 4 classes of image prediction goals: american shorthair, british shorthair, exotic short hair and scottish fold. 
- Selected 3 CNN backbones to fine tuning by adjust hyperparameters and change classifier layer of models with our custom data set.

## Assumption: <a name="assumption"></a>


## Data Set Overview: <a name="dataset"></a>
This data was collected via pinterest website. Stored each class of images in a separate folder and all images are kept as jpeg format.

![image](https://user-images.githubusercontent.com/80414593/196958981-14128603-77a2-416a-a048-63ef15ff40ae.png)

There are a total of 775 images divided into 4 classes as follows:
1. American shorthair : 190 images
2. British shorthair : 185 images
3. Exotic short hair : 205 images
4. Scottish fold : 195 images

### Prepare Data
Data spliting -> Train 80%, Test 10%, Validation 10%

![image](https://user-images.githubusercontent.com/80414593/196958239-74da6aee-fd42-45d4-ab7e-124770b51674.png)

## Data Augmentation: <a name="augment"></a>
We augment 2 time on train set with tensorflow.keras.Sequential with layers as follow:
1. layers.RandomFlip("horizontal",input_shape=(img_height, img_width, 3))
2. layers.RandomRotation(0.1)
3. layers.RandomZoom(0.1)
4. layers.RandomBrightness(factor=0.3,value_range=(0, 255), seed=64)


The total number of images after we augmented:

![image](https://user-images.githubusercontent.com/80414593/196957976-45e6b369-4ca0-46e5-beb9-566c5a9cf825.png)

Example image after augment:

![image](https://user-images.githubusercontent.com/80414593/196961061-5875f4fa-2825-46d8-98f9-fa4c2febe11b.png)


## Convolutional Neural Network BackBones: <a name="bb"></a>
1. Resnet50 
2. Mobilenet small
3. VGG16

## Network Architecture: <a name="architecture"></a>


## Training and Fine tuning: <a name="finetuning"></a>

## Discussion: <a name="discussion"></a>

## Conclusion: <a name="conclusion"></a>

## Reference: <a name="reference"></a>
### Library

### Version

### Set Seed
np.random.seed(1234)
tf.random.set_seed(5678)

