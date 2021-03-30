# **Traffic Sign Recognition** 

## Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/8.jpg "Traffic Sign 1"
[image2]: ./images/11.jpg "Traffic Sign 2"
[image3]: ./images/12.jpg "Traffic Sign 3"
[image4]: ./images/17.jpg "Traffic Sign 4"
[image5]: ./images/25.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/RyanFDU/P3-Traffic-Sign-Classifier/blob/f10b3e44afe76eb4927c8969ed8421b2716b6430/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the `np.shape` to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I made an exploratory visualization of the data set which were shown in my jupyter as a bar chart.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Following this paper [Sermanet, LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) I employed three main steps of feature preprocessing:

1) *each image is converted from RGB to YUV color space, then only the Y channel is used.* This choice is shown in the cited paper how it leads to the best performing model. This is slightly counter-intuitive, but it's quite similar as looking to the grayscale image.

2) *contrast of each image is adjusted by means of histogram equalization*. This is to mitigate the numerous situation in which the image contrast is really poor.

3) *each image is centered on zero mean and divided for its standard deviation*. This normalization has beneficial effects on the gradient descent performed by the optimizer.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final architecture is a relatively shallow network made by 4 layers. The first two layers are convolutional, while the third and last are fully connected. Following [Sermanet, LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) the output of both the first and second convolutional layers are concatenated and fed to the following dense layer. In this way we provide the fully-connected layer visual patterns at both different levels of abstraction. The last fully-connected layer then maps the prediction into one of the 43 classes.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| ReLu					|				  				|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Drop out					|				keep_prob = 0.5					|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x128 	|
| ReLu					|				  				|
| Max pooling	      	| 2x2 stride,  outputs 8x8x128 				|
| Drop out					|				keep_prob = 0.5					|
| Concat					|				Conbine two Drop-out outputs			|
| Fully connected		|    		outputs 1x1x64					|
| Drop out					|				keep_prob = 0.5					|
| Fully connected		|    		outputs 43 classes					|
| Softmax				|   									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer because it faster convergence and reduced oscillation and the parameters were tuned as below after several attempts:

* batch size: 128
* epochs: 30
* batches_per_epoch = 5000
* learning rate: 0.001
* dropout keep probability: 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final architecture is a relatively shallow network made by 4 layers. The first two layers are convolutional, while the third and last are fully connected. Following [Sermanet, LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) the output of both the first and second convolutional layers are concatenated and fed to the following dense layer. In this way we provide the fully-connected layer visual patterns at both different levels of abstraction. The last fully-connected layer then maps the prediction into one of the 43 classes.

To get additional data, I use the `ImageDataGenerato` class provided in the Keras library so that I can make the data augmentation online. In this way, images are randomly rotated, zoomed and shifted in a narrow range, thus creating some variety in the data.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.961
* test set accuracy of 0.934 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![class 8][image1] ![class 11][image2] ![class 12][image3] 
![class 17][image4] ![class 25][image5]

The fourth image might be difficult to classify because its bad quanlity. The contrast is pretty low and it even has some jitters.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (120km/h)      		| Speed limit (120km/h)   									| 
| Right-of-way at the next intersection    			| Right-of-way at the next intersection										|
| Priority road					| Priority road											|
| No entry	      		| Traffic signals					 				|
| Road work			| Road work      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is in the last third cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Right-of-way at the next intersection sign (probability of 0.98), and the image does contain a Right-of-way at the next intersection sign. The top five soft max probabilities were:

| Probability         	|     Prediction       					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			|     Right-of-way at the next intersection						| 
| .02     				|   Beware of ice/snow					|
| .00					|   Dangerous curve to the left						|
| .00	      			|    Slippery road			 				|
| .00				    |   Road work  							|

Other images show quite the same result, except for the No entry sign which was put in the wrong class possibly due to its low contrast and jitters.
