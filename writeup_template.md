# **Behavioral Cloning** 

## Writeup Template

### gitYou can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/center_lane_driving.jpg "CenterDriving"
[image3]: ./examples/Recovery1.jpg "Recovery Image"
[image4]: ./examples/Recovery2.jpg "Recovery Image"
[image5]: ./examples/Recovery3.jpg "Recovery Image"
[image6]: ./examples/NormalImg.jpg "Normal Image"
[image7]: ./examples/FlipedImg.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128.

The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The training includes full track trips and recovery driving to reduce overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was first to use LeNet's architecture.

This did okay but was very jittery and the steering didn't seem human like (smooth and gentle).

The next trail was Nvidia's architecture.

I just sliced off the last layer and pluged it into my classifier which is called transfer learning.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the training data so that there were better examples and that the model wasn't learning off of bias data or "bad driving". 

The final step was to run the simulator to see how well the car was driving around track one.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (clone.py lines 61-76) consisted of a convolution neural network with the following layers and layer sizes:
* Normalize (160x320x3)
* Crop Image 70 from the top 20 from the bottom
* Convolution 5x5x25, subsample = (2x2)
* activation relu
* Convolution 5x5x36, subsample = (2x2)
* activation relu
* Convolution 5x5x48, subsample = (2x2)
* activation relu
* Convolution 3x3x64
* activation reul
* Convolution 3x3x64
* activation reul
* Flatten
* Fully connected (100)
* Fully connected (50)
* Fully connected (10)
* Fully connected (1)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving and some outer to center lane driving. Since I wasn't such a good driver I ended up using Udacity's data that they provided. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would increase training and prevent overfitting. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 38572 number of data points. I then preprocessed this data by:
* Adding/Subtracting driving angle to left and right camera respectfully.
* Flipping the images.
* Normalizing.
* Finally cropping all of the images to get rid of noise background pixels.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.
