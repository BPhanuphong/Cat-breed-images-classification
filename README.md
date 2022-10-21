# Cat-breed-images-classification :smiley_cat:
## README-File
-
-
-

## Executive Summary:


## :cat2:  Table of Contents:
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Augmentation](#augment)
4. [CNN Backbones](#bb)
5. [Training and Fine-Tuning](#finetuning)
6. [Result](#result)
7. [Discussion](#discussion)
8. [Conclusion](#conclusion)
9. [Reference](#reference)

## Introduction: <a name="introduction"></a>
- The purpose of this project is to train an image classification (single-label) with our custom dataset (Cat breed images) by pre-trained CNN models. By, cat breed images consist of 4 classes of image prediction goals: **``American shorthair, British shorthair, Exotic shorthair and Scottish fold``**.
- The pre-trained models we selected 3 CNN backbones **``(Resnet50, Mobilenet small, VGG16)``** to fine-tuning by adjusting hyperparameters, unfreeze layers and changing the classification layer of models with our custom dataset, After fine-tuned we compare 3 model with original pre-trined CNN models(no fine-tuning).

## Dataset Overview: <a name="dataset"></a>
This data was collected via pinterest website. Stored each class of images in a separate folder and all images are kept as jpeg format.

![image](https://user-images.githubusercontent.com/80414593/196963734-1461e440-6c55-4321-9e2e-c528bb4be783.png)

There are a total of 775 images divided into 4 classes as follows:
1. Exotic shorthair : 205 images
2. Scottish fold : 195 images
3. American shorthair : 190 images
4. British shorthair : 185 images

Sample image of each class:
1. Exotic shorthair 

![image](https://user-images.githubusercontent.com/80414593/196993705-cb5e9f35-3dbe-43ab-acc6-e0ce71d54e33.png)

2. Scottish fold 

![image](https://user-images.githubusercontent.com/80414593/196995495-2fcc2dff-2cdf-4b43-aa63-f335d893f80d.png)

3. American shorthair

![image](https://user-images.githubusercontent.com/80414593/196995623-3d604fa3-e930-4b66-b1b1-f77040d755eb.png)

4. British shorthair 

![image](https://user-images.githubusercontent.com/80414593/196994093-bfed1356-d77b-4c9d-8c72-ceabac9b00ca.png)


### Prepare Data
- `` Resize image shape`` -> 224, 224 
- `` Data spliting`` -> Train 80%, Test 10%, Validation 10%

<!--- We will fit model with train set and test set. For validation set is used to evaluate accuracy's model. --->

![image](https://user-images.githubusercontent.com/80414593/197229829-0eacbfea-8abb-4cfb-9526-3dbc5654a0b0.png)


## Data Augmentation: <a name="augment"></a>
After we rescale pixel of all images and split in to 3 set, We do the data augmentation on the trian to improve performance and outcomes of the models by forming new and different examples. In this process augmented images will be stored as a new train dataset. We will augment on train set with 2 medthods as follow:

1. Augment 2 times on train set with tensorflow.keras.Sequential with layers as follow:
  - layers.RandomFlip("horizontal",input_shape=(img_height, img_width, 3))
  - layers.RandomRotation(0.1)
  - layers.RandomZoom(0.1)
  - layers.RandomBrightness(factor=0.3,value_range=(0, 255), seed=64)

Example of images after augment by method 1:

![image](https://user-images.githubusercontent.com/80414593/196961061-5875f4fa-2825-46d8-98f9-fa4c2febe11b.png)


2. Hue Augmentation: We augment 1 time on train set to change temperature of image with tf.image.stateless_random_hue(x_train, 0.5, seed)

Example of images after hue augmentation:

![image](https://user-images.githubusercontent.com/80414593/196975525-2386a39e-7cc5-4047-b37b-e62ff725627b.png)

The total number of images after we augmented:

![image](https://user-images.githubusercontent.com/80414593/196957976-45e6b369-4ca0-46e5-beb9-566c5a9cf825.png)



## Convolutional Neural Network BackBones: <a name="bb"></a>
We load the models with imagenet weight, **excluding** the latter part regarding the classifier
1. Resnet50 
2. Mobilenet small
3. VGG16

![image](https://user-images.githubusercontent.com/80414593/197187459-c813dfba-6fb6-4405-990e-2634d026933e.png)

*Ref: https://keras.io/api/applications/*

:stars: The image show detail of each pre-trained model, We selected all of the different sized models so that we can compare how each is, when applied to our dataset, whether or not the size of the model affects the accuracy. 


## Training and Fine-Tuning: <a name="finetuning"></a>
### Label of classes
- `` Exotic shorthair : 0 ``
- `` Scottish fold : 1 ``
- `` American shorthair : 2 ``
- `` British shorthair : 3 ``

### Strategy Fine-tuning:
Individually fine-tuning for each model for the best accuracy.

First, We fine-tuning the pretrained model by *adjusting hyperparameters* of each models. Once the best hyperapameters has been obtained from the experiment, We will *Unfreeze layer* of pre-trained models to train the weights of the top layers. Next We will gradually modify the layers on the *classifier's part*.

Range of hyperparameters that we adjust :
- `` Optimizer`` : [Adam]
- `` Learning Rate`` : [0.0000001, 0.000001, 0.00001, 0.0001, 0.00015, 0.0002, 0.001, 0.0025, 0.004, 0.0045, 0.005, 0.006, 0.007, 0.0075, 0.008, 0.01, 0.1]
- `` Batch size`` : [1, 60, 64, 100, 120, 128, 136, 150, 186, 200, 256, 300, 500]
- `` Epoch`` : [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 35, 40, 50, 100, 150, 200, 250, 300, 400, 500, 550, 600]
- `` Loss Function`` : [Sparse_catergoricL_crossentropy]

Classifier:
- `` Dense`` :[256,512,1024]
- `` Regularization`` :[Dropout(0.5), Dropout(0.75), Dropout(0.85)]

### Compare Model: Fine Tuned CNN Model vs Original Pre-Trained CNN Model
:speech_balloon: *Original Pre-Trained CNN Model is the model that we load with IMAGENET weight and make predictions on our cat breed dataset.*

We compare models that we fine-tuned with their original models, give 4 input to model (1 image of each classes)
### Resnet50

![image](https://user-images.githubusercontent.com/80414593/197201125-10d57379-19e2-4296-b771-123bd93af0f3.png)

## :triangular_flag_on_post: Result<a name="result"></a>
Result of the fine-tuned 3 models will compare with base model of thier models.(*base model: Pre-trained model before fine-tuning*)
### Resnet50v2 
Hyperparameter's fine-tuned
- `` Optimizer`` : Adam
- `` Learning Rate`` : 0.0001
- `` Batch size`` : 256
- `` Epoch`` : 50

UnFreeze layer -> [176:189]
```
- conv5_block3_3_conv 
- conv5_block3_out 
- post_bn 
- post_relu 
```
Fine-tuned classifier
```
- Flatten
- Dense 512
- Dropout 0.75
- Dense 4
```

Result

## Discussion: <a name="discussion"></a>


## Conclusion: <a name="conclusion"></a>

## Reference: <a name="reference"></a>
### GPU
GPU 0: Tesla T4 (UUID: GPU-0c0e16f8-6133-6dc4-d1b4-843e49281c9e)

### Library Version
Python 3.7.15

NumPy 1.21.6

TensorFlow 2.9.2

Sklearn 1.0.2

Matplotlib 3.2.2

### Set Seed
np.random.seed(1234)

tf.random.set_seed : [5678, 8753, 1947, 2012, 8289, 1238]

