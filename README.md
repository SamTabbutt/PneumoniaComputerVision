# PneumoniaComputerVision
Kaggle dataset machine vision application

I found a pneumonia x-ray dataset on Kaggle and decided to tackle the problem with tensorflow/keras neural networks. Due to the size of the dataset,
I have added the folder to .gitignore. The dataset can be found [here](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

## Provided data

The dataset consists of 5216 training images, split 25% x-ray images of pneumonia-negative chests, and 75% x-ray images of pneumonia-positive chests. The set of positive-pneumonia images is sub-divided as baterial pneumonia cases and viral pneumonia cases. This split is 65% bacterial and 35% viral.

Accompanying the training data is a validation set containing 16 images with a 50/50 split of negative and positive x-rays. Each of the positive images in the validation set are baterial infections.

Finally there is a labeled test set consisting of 633 images split 62:38::positive:negative with 62:38::bacterial:viral for the positive images.

## The Problem

Train a machine learning model using the training and validation datasets and apply the model to the testing set, yielding an accuracy of the trained model. The goal is to maximize the testing accuracy.

## Solution

In identifying and solving the problem, there are three processes necesarry:
- [Unpack and preprocess data](https://github.com/SamTabbutt/PneumoniaComputerVision/blob/master/README.md#preprocessing)
- Define and train ML model
- Run model on test set

### Preprocessing
I started by addressing the unpacking and preprocessing of the datasets. The module ```preprocessing.py``` contains an object imageSet, which can be called to create a full numpy array of preprocessed images with a corresponding label numpy array which contains the target classification for each image in the array. 

I started by processing the positive-pneumonia x-rays with separate labels for 'bateria' and 'virus', giving three possible labels for the output:
- NORMAL: 0
- bacteria: 1
- virus: 2

The imageSet object takes three arguments: 
- labeled (*bool*): Boolean of whether or not the input folder contains labeled photos
- path (*string*): the path to the folder containing the photos
- targetShape (*tuple*): the shape to resize each image to (h,w), to produce a preprocessed numpy array of shape (N,h,w,1) where N is the number of photos in the folder

The imageSet object has two notable fields:
- self.X (*numpy array*): normalized numpy array of shape (N,h,w,1) where N is the number of photos in the folder. self.X has a range of [0,1]

![formula](https://render.githubusercontent.com/render/math?math=X%20\equiv%20\{(i,\overline{\rm f},1):\overline{\rm%20f}\in[0,1]^{h\times%20w}\land%20i<N\})
- self.y (*numpy array*): numpy array containing labels of shape (N,) where N is the number of photos in the folder. self.y is defined in a discrete range over the set {0,1,2} with each number corresponding to the label defined above. 

*NOTE*: For each image in the folder 'path', there exists an element in self.X at position n that is the normalized and resized representation of the image with the corresponding label at the nth position of self.y

*NOTE*: For user interfacing with the terminal, I used [this script](https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console) to provide a status of the unpacking of the images. It is in the path ```<working_dir>/misc/progressBar.py```

### Defining and training model

The process for defining and training the model are in the module ```kerasPipeline.py```. The model is defined as a class child to ```tf.keras.Model``` and is represented by the following chart:
![text](https://github.com/SamTabbutt/PneumoniaComputerVision/blob/master/misc/disp/ModelInit.png)

The initial run yielded the following run-time statistics:
![text](https://github.com/SamTabbutt/PneumoniaComputerVision/blob/master/misc/disp/InitStats.JPG)

In summary, the semmingly trivial model was able to evaluate the test set with 100% accuracy.
