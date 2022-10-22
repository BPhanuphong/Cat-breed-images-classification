# Cat-Breed-Images-Classification using Pre-Trained CNN Model :smiley_cat:
## README-File
- ``data set`` : A folder that contains all images used for training in the model are separated by the cat's breeds folder.
- ``resnet50v2_Final.ipynb`` : Final model after fine-tuning the Reanet50v2 backbone.
- ``mobilenetv3_small_V3_Final.ipynb`` : Final model after fine-tuning the MobileNetv3Small backbone. 
- ``vgg16_Final.ipynb`` : Final model after fine-tuning the VGG16 backbone. 
- ``Final Result.xlsx`` : The results of all the fine-tuning experiments.


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

## :round_pushpin: Introduction: <a name="introduction"></a>
- The purpose of this project is to train an image classification (single-label) with our custom dataset (Cat breed images) by pre-trained CNN models. By, cat breed images consist of 4 classes of image prediction goals: **``American shorthair, British shorthair, Exotic shorthair and Scottish fold``**.
- The pre-trained models we selected 3 CNN backbones **``(Resnet50v2, MobileNetv3Small, VGG16)``** to fine-tuning by adjusting hyperparameters, unfreeze layers and changing the classification layer of models with our custom dataset, After fine-tuned we compare 3 model with original pre-trined CNN models(no fine-tuning).

## :clipboard: Dataset Overview: <a name="dataset"></a>
This data was collected via several websites by ourselves. Stored each class of images in a separate folder and all images are kept as jpeg format.

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
- `` Data spliting (random split)`` -> Train 80%, Test 10%, Validation 10%

<!--- We will fit model with train set and test set. For validation set is used to evaluate accuracy's model. --->

![image](https://user-images.githubusercontent.com/80414593/197320405-00a918ee-5b00-466e-83d5-06c1e3fd59a6.png)

## :mage:  Data Augmentation: <a name="augment"></a>
After we rescale the pixel of all images and split them into 3 sets, We do the data augmentation on the train to improve the performance and outcomes of the models by forming new and different examples. In this process, augmented images will be stored as a new train dataset. We will augment on train set with 2 methods as follow:

1. Augment 2 times on train set with tensorflow.keras.Sequential with layers as follow:
  - layers.RandomFlip("horizontal",input_shape=(img_height, img_width, 3))
  - layers.RandomRotation(0.1)
  - layers.RandomZoom(0.1)
  - layers.RandomBrightness(factor=0.3,value_range=(0, 255), seed=64)

Example of images after augment by method 1:

![image](https://user-images.githubusercontent.com/80414593/197326555-c6ba742d-5647-4be7-8be5-c50775b75941.png)

2. Hue Augmentation: We augment 1 time on train set to change temperature of image with tf.image.stateless_random_hue(x_train, 0.5, seed)

Example of images after hue augmentation:

![image](https://user-images.githubusercontent.com/80414593/197326485-d1337da5-7a68-41f4-ad4a-8251156d5161.png)


The total number of images after we augmented:

![image](https://user-images.githubusercontent.com/80414593/197344505-ff4957f1-346b-49a8-91ee-976aa75ef7f7.png)

## Convolutional Neural Network BackBones :bone: : <a name="bb"></a>
The data is ready to use, Now we will start pre-trained model part. We load the models with IMAGENET weight, **excluding** the latter part regarding the classifier because we will build the classifier part by ourselves to make a classifier fit with our cat dataset. 

1. Resnet50v2
2. MobileNetv3Small
3. VGG16

![image](https://user-images.githubusercontent.com/80414593/197187459-c813dfba-6fb6-4405-990e-2634d026933e.png)

*Ref: https://keras.io/api/applications/* <a name="kerastable"></a>

:stars: The image show detail of each pre-trained model, We selected all of the different sized models so that we can compare how each is, when applied to our dataset, whether or not the size of the model affects the accuracy. 


## Training and Fine-Tuning :crystal_ball:: <a name="finetuning"></a>

### :label: Label of classes
- `` Exotic shorthair : 0 ``
- `` Scottish fold : 1 ``
- `` American shorthair : 2 ``
- `` British shorthair : 3 ``

### :mag_right:  Strategy Fine-tuning:
Individually fine-tuning for each model for the best accuracy.

First, We fine-tuned the pre-trained model by *adjusting the hyperparameters* of each model. Once the best hyperparameters have been obtained from the experiment, We will *Unfreeze the layer* of pre-trained models to train the weights of the top layers. Next, We will gradually modify the layers on the *classifier's part*.

Range of hyperparameters that we adjust :
- `` Optimizer`` : [Adam]
- `` Learning Rate`` : [0.0000001, 0.000001, 0.00001, 0.0001, 0.00015, 0.0002, 0.001, 0.0025, 0.004, 0.0045, 0.005, 0.006, 0.007, 0.0075, 0.008, 0.01, 0.1]
- `` Batch size`` : [1, 60, 64, 100, 120, 128, 136, 150, 186, 200, 256, 300, 500]
- `` Epoch`` : [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 35, 40, 50, 100, 150, 200, 250, 300, 400, 500, 550, 600]
- `` Loss Function`` : [Sparse_catergoricL_crossentropy]

Classifier:
- `` Dense`` :[256,512,1024]
- `` Regularization`` :[Dropout(0.5), Dropout(0.75), Dropout(0.85)]

### Model Architecture after Fine-Tuning
![image](https://user-images.githubusercontent.com/80414593/197343049-a957fcee-2fec-4413-bf3b-ec4b1f98a848.png)

### :dart: Compare hyperparameters of all models: Resnet50v2, MobileNetv3Small, VGG16
![image](https://user-images.githubusercontent.com/80414593/197344216-f7055e80-b93c-4813-bc7c-18d651eb5fbd.png)


## :triangular_flag_on_post: Result<a name="result"></a>
The result of the fine-tuned 3 models will compare with the base model of their models. (*base model: Pre-trained model before fine-tuning*) 

*Note:* The results of the models to compare the performance were obtained by running the models with 6 different initial weights ([random seed](#seed)), Then showing the result in the form of MEAN+-SD.

### :crossed_swords: 1. Compare Model: Base Model vs Fine-Tuned Model

### :dart: Resnet50v2 

![image](https://user-images.githubusercontent.com/80414593/197329572-9cd8f4db-6f7d-46b8-8af5-ca289b881122.png)

### :dart: MobileNetv3Small 

![image](https://user-images.githubusercontent.com/80414593/197329705-47d84813-7166-4124-be4a-b1680c200b22.png)

### :dart: VGG16 

![image](https://user-images.githubusercontent.com/80414593/197329727-04c3a3b8-7a15-4137-9fc8-ddca1e462ccc.png)



### 	:crossed_swords: 2.  Compare Model: Original Pre-Trained CNN Model vs Fine-Tuned Model
:speech_balloon: *Original Pre-Trained CNN Model is the model that we load with IMAGENET weight and make predictions on our cat breed dataset.*

We compare models that we fine-tuned with their original models, and give 4 inputs to model (1 image of each classes). The figure below shows the probabilistic prediction results of the class below the image obtained from the original model and our fine-tuned model.

### :dart: Resnet50v2
![image](https://user-images.githubusercontent.com/80414593/197326126-c869a0c5-a295-4b40-a8a4-7d0cbc22b10e.png)

### :dart: MobileNetv3Small
![image](https://user-images.githubusercontent.com/80414593/197328657-603822fa-6f85-45a0-a50e-772a0c8e6768.png)

### :dart: VGG16
![image](https://user-images.githubusercontent.com/80414593/197326212-09054d38-1bea-43db-b615-e06dd312acf9.png)






## :page_facing_up:	 Discussion: <a name="discussion"></a>
### 1. Base Model (Before fine-tuning) vs Fine-Tuned Model
![image](https://user-images.githubusercontent.com/80414593/197329332-ec350059-0de5-45ff-a430-bb65d87bd674.png)

---
<!--- 1 --->
![image](https://user-images.githubusercontent.com/80414593/197344270-1de47094-d125-48da-a068-55ebe397024d.png)
- Resnet50v2 model and MobileNetv3Small model have improved accuracy after fine-tuning and the training time is also increasing.
- VGG16 model the accuracy is not much better after fine-tuning and the training time is take so long because VGG16 is a sequential model.
---
<!--- 2 --->
![image](https://user-images.githubusercontent.com/80414593/197344969-ad8a2be2-ee35-4f4c-b3f6-50e02ef5f89e.png)
- Resnet50v2 model has the highest accuracy gain compared to other models and has a higher percentage of standard deviation.
- VGG16 model has a slightly decrease in accuracy.For the standard deviation VGG16 had the lowest increase compared to the other models.
- All models after fine-tuned will have at least a 40% standard deviation increase.
---
<!--- 3 --->
![image](https://user-images.githubusercontent.com/80414593/197345510-a7fdb07e-debd-4f40-8592-1d2605d7dec9.png)
- The percentage increase of training time in Resnet50v2 is the lowest but training time of Resnet50v2 is the most swinging.
- As for MobileNetv3Small, the training time is less swinging compared to the other two models.
---
<!--- 4 --->
![image](https://user-images.githubusercontent.com/80414593/197345904-3fe24f43-43a9-4aee-a1a7-1c0eb61c603a.png)
- Resnet50V2 has the most all parameters compared to others because ResNet50v2 has the most feature extractors. And when we connect to the classifier layer we get 70 million parameters. This is one of the reasons why the Resnet50V2 is the most accurate.

- VGG16 has the most training time. But the parameters are less than Resnet50V2. Our assumptions about this may be due to a model architecture that slows training based on [Keras training tables](#kerastable).

---
<!--- 5 --->
![image](https://user-images.githubusercontent.com/80414593/197346194-a9a27ed7-14d2-4cad-8d7e-7040b28838ca.png)

---
<!--- 6 --->
![image](https://user-images.githubusercontent.com/80414593/197346095-90dc058d-d1fd-46d5-87df-d6d433d2a428.png)

From our study and experiment, we found that factors affecting model results may compose of
- The proportion of features extractor of the original model
- Model structure.
- Layer and parameter that add to the model.
Moreover from information from this experiment, we assume that the reason why Resnet50V2 gives the best result from this experiment is that we use only the feature extractor of each backbone to connect with classifiers that we build by ourselves and due to the high proportion in feature extractor in ResNet50v2 Model so it gives us the better result that other 2 model.
Furthermore, due to the lowest proportion of feature extractor of VGG16 Model and the classifier that we add to make the whole parameter smaller than the original one, these will cause the lowest result and performance of its fine-tuned model compared with the other 2 models. However, if we have a chance to do more experiments and studies, we will get more evidence to support or refute these assumptions.



### 2. Original Pre-Trained CNN Model (with IMAGENET weight) vs Fine-Tuned Model
- Original pertained model has trained with specific class that our data class not included. On MobileNetV3Small and Vgg16 has predicted black Exotic shorthair as space heaters while Resnet50v2 predicted as Schipperke (Black dog). On all fine-tune model can predict correctly on all. breeds cats.






## :trophy:  Conclusion: <a name="conclusion"></a>
### 1. Base Model (Before fine-tuning) vs Fine-Tuned Model
- 

### 2. Original Pre-Trained CNN Model (with IMAGENET weight) vs Fine-Tuned Model
- Resnet50v2 pre-trained model has a background in knowledge of cats than the other 2 models because the Resnet50v2 pre-trained model predicted the class closest to the cat it predicted the black dog breed, which is also considered a class that looks close to the cat. (These conclusions provide support to the conclusion of topic 1)


## Reference: <a name="reference"></a>
### Data Source
- **https://www.pinterest.com/**
```
- Keyword: exotic shorthair cat, Date: 2-6 October 2022
- Keyword: scottish fold, Date: 2/6 October 2022
- Keyword: american shorthair, Date: 2/8 October 2022
- Keyword: Brithish shorthair cat, Date: 8/9/15 October 2022
```
- https://www.petfinder.com/, Date: 2-6 October 2022
- https://www.britishcattery.com/, Date: 15 October 2022




### GPU
- GPU 0: Tesla T4 (UUID: GPU-0c0e16f8-6133-6dc4-d1b4-843e49281c9e)

### Library Version
- Python 3.7.15
- NumPy 1.21.6
- TensorFlow 2.9.2
- Sklearn 1.0.2
- Matplotlib 3.2.2

### Set Seed
- np.random.seed(1234)
- tf.random.set_seed<a name="seed"></a> : [5678, 8753, 1947, 2012, 8289, 1238]

### Group Member
| Member        | % Contribution | Main Responsibility |
| :------------ |:--------------:|--------------------:|
| 6310422101    |       25%     | Collect data, Fine-tune MobileNetv3Small model       |
| 6310422102    |       25%     | Collect data, Fine-tune VGG16 model           |
| 6510400003    |       25%    | Main coding, Fine-tune Resnet50v2 model       |
| 6510400004    |       25%      | Collect data, Support coding,  Document Report  |


### About Project
This project is a part of the

- Subject: **DADS7202 Deep Learning :star:**
- Course: **Data Analytics and Data Science (DADS)**
- Institution: **National Institute of Development Administration (NIDA)**
