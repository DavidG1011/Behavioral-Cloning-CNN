**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Example_Images/left.jpg "Left"
[image2]: ./Example_Images/center.jpg "Center"
[image3]: ./Example_Images/right.jpg "Right"


***[Rubric](https://review.udacity.com/#!/rubrics/432/view) Points***
---

Files included:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 


***Running The Code***
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

The model.py file contains the code for training and saving the convolution neural network.

---

**Training Strategy**

---
***Preprocessing and Normalization***

1. To start, I used Keras to crop my images. 
```Cropping2D(cropping=((70, 25), (0, 0)))```
This removes the top 70 pixels from the image, as well as the bottom 25. This cuts off excess "useless" data and helps the model to focus and train only on the curvature of the road, as opposed to trees or the hood of the car. 

2. Secondly, I used the formula: ```x/255.0 - 0.5``` to normalize my image set. This helps to keep the images similar and reduce variances. 
---

My model architecture is based off the NVIDIA End To End Learning For Self driving cars paper, documented [Here](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)


***Final Model Architecture - NVIDIA (Link Above)***

After fiddling with simple architectures and trying to make the most of them-- and nothing cutting it in terms of the result I wanted, I decided to use a tried and true architecture. I settled on the NVIDIA architecture because of its relative simplicity, and frankly, because it works favorably for the end goal of keeping a car on the track.


|Layer           | Details                          |
|:--------------:|:--------------------------------:|
|Normalization | x/255.0 - 0.5 , Input shape: 160x320x3|
|Convolution 24x5x5 | 2x2 subsample, ReLU activation| 
|Convolution 36x5x5 | 2x2 subsample, ReLU activation|
|Convolution 48x5x5 | 2x2 subsample, ReLU activation|
|Convolution 64x3x3 | ReLU activation|
|Convolution 64x3x3 | ReLU activation|
|Flat Convolution| ... |
|Fully Connected | Dense (100) | 
|Dropout Layer | 25% Probability |
|Fully Connected | Dense (50) | 
|Dropout Layer | 25% Probability |
|Fully Connected | Dense (10) | 
|Dropout Layer | 25% Probability |
|Fully Connected | Dense (1) |


***Reducing Overfitting In The Model***

 To reduce overfitting, I introduced dropout layers after each fully connected layer. In running the model, this seems to be effective.
 
 Some empirical data:
 
 ***Output Snippet With Dropout Layers:***
 ```Epoch 3/10
25s - loss: 0.0177 - val_loss: 0.0161
Epoch 4/10
25s - loss: 0.0167 - val_loss: 0.0159
```

***Output Snippet Without Dropout Layers***
```Epoch 3/10
27s - loss: 0.0107 - val_loss: 0.0151
Epoch 4/10
26s - loss: 0.0127 - val_loss: 0.0169
```

As can be seen above: without dropout layers, the training loss would be much lower than the validation loss, as well as the validation loss increasing with each epoch, as opposed to decreasing as it should.


***Model Parameter Tuning***

The model used an adam optimizer, so the learning rate was not tuned manually.


***Training Data Selection***

The training data used was the sample set provided by Udacity, linked [Here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip). I found that this data was more "complete" and produced better results than the set I obtained by running the simulator myself. When using training data i obtained myself, The car had a tendency to be far too close to the outer edge of the track, often running off the track. 


***Multiple Camera Angles***

As detailed in my ```model.py``` code, I used all camera angles to increase the amount of training data used and make turning easier. I found this to be one of the most important changes to my network. It was a night and day transformation in my turning performance. 

***Example Images***

Here is a captured set from the left, center, and right cameras at the same moment on the track:

Left:

![alt text][image1]

Center:

![alt text][image2]

Right:

![alt text][image3]


***The Training Strategy: Trial And Error... Plenty of Error***

* 1. The first step in designing this model was to obtain a large set of training data. To do this, I first set off to obtain the data from my own driving on the track. I feel I did well enough at staying close to the center of the track to give the optimal training data to my model.

* 2. If it was good enough driving was not yet apparent to me, because my model was far too simple to utilize even the best driver data in the world. My first iteration of my model architecture probably shouldn't have even been used-- but I wanted a good baseline for how complex I needed my model to be to achieve the outcome of staying on the track effectively. My first iteration was simply:
```


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
