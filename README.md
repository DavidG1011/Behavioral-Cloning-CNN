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
* track1.mp4 showing a successful lap of the track in autonomous mode


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

The model is then compiled with the following paramaters:

* Loss: MSE (Mean Squared Error)
* Optimizer: Adam

And trained with the following paramaters:

* Validation Set: 20% of training set
* Shuffled: True
* Epochs: 10 -- Has been run for more but 10 is fine
* Early Stop: Stops if validation loss hasn't improved for 2 epochs


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


Training data specifications:
Number of training examples:  15429
Number of validation examples:  3857
Number of testing examples:  4822


***Multiple Camera Angles***

As detailed in my ```model.py``` code, I used all camera angles to increase the amount of training data used and make turning easier. I found this to be one of the most important changes to my network. It was a night and day transformation in my turning performance using it vs. only using the center camera. And this is to be expected, I was using 3x the amount of training data. 

***Example Images***

Here is a captured set from the left, center, and right cameras at the same moment on the track:

Left:

![alt text][image1]

Center:

![alt text][image2]

Right:

![alt text][image3]


***The Training Strategy: Trial And Error... Plenty of Error***

* The first step in designing this model was to obtain a large set of training data. To do this, I first set off to obtain the data from my own driving on the track. I feel I did well enough at staying close to the center of the track to give the optimal training data to my model.

* The quality of the training data was not yet apparent to me, because my model was far too simple to utilize even the best driver data in the world. The first iteration of my model architecture probably shouldn't have even been used-- but I wanted a good baseline for how complex I needed my model to be to achieve the outcome of staying on the track effectively. My first iteration was simply:
```
model.add(Conv2D(32, 5, 5, activation="relu", input_shape=(160, 320, 3)))
model.add(Flatten())
model.add(Dense(1))
```
Now where's my tech award?

* This obviously wasn't going to cut it, it was underfitting due to lack of convolutions, and the driving data showed just how bad this model truly was. The car swerved immediately off the track. This model was also enacted before i used multiple camera angles, because I wanted to measure basline performance expectations based on only one input stream of images. 

* The next iteration introduced more convolutional layers and fully connected layers. 
```
model.add(Conv2D(32, 5, 5, activation="relu", input_shape=(160, 320, 3)))
model.add(Conv2D(32, 5, 5, activation="relu"))
model.add(Flatten())
model.add(Dense(28))
model.add(Dense(1))
```
I was happy with the reduction in overfitting and the slightly better driving performance, it could now go a whole 10 feet before running off the road and into a tree.

* Next I added a group of two more convolutions to see if i could squeeze some more data out of it
```
model.add(Conv2D(32, 5, 5, activation="relu", input_shape=(160, 320, 3)))
model.add(Conv2D(32, 5, 5, activation="relu"))
model.add(Conv2D(64, 5, 5, activation="relu"))
model.add(Conv2D(64, 5, 5, activation="relu"))
model.add(Flatten())
model.add(Dense(28))
model.add(Dense(1))
```
It started to look somewhat like a model. The validation loss was now in the "ok" range and 
the car was able to go mostly straight up until the first turn where it would promptly lose control.

* It was at this point that I decided it may be better to start thinking about an already established model, which is when I made the transition to the one detailed in the NVIDIA End to End Driving paper.

* In using that model, I found that the model performed well, as expected, but still introduced some overiftting with the amount of data I had. from there I implemented a dropout layer with 25% probability after each fully connect layer like so:
```
...
model.add(Dense(100))
model.add(Dropout(0.25))
model.add(Dense(50))
model.add(Dropout(0.25))
model.add(Dense(10))
model.add(Dropout(0.25))
model.add(Dense(1))
```
With this final model-- detailed above, the vehicle is able to drive autonomously around the track without leaving the road. I have even put the model through its paces a bit. My findings on that can be found below:

---

**Putting the Model Through Its Paces: Generalizing**

---

**Navigating back to the center:**

I have done many test cases where I put the car manually to the outer edge of the track, and in the majority of cases, it can successfully navigate back to the center of the road-- or into the "safe to drive" zones of the road. It has slight difficulty if it goes to the edge during a turn, however.

**Reverse Lap:**

Putting the car in a clockwise manner around the track, it can still just as easily navigate the entirety of it.

---


***Conclusion:***

I am happy with the way that the model performs, and I hope to do something like this again in the future.
