# Traffic Sign Recognition
## Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

### Writeup / README

*Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report.*

The submission includes the project code.

This report is the write-up.

### Data Set Summary & Exploration
*Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.*

![data shapes](https://github.com/cswksu/trafficSignClassifierUdacity/blob/master/images/dataSize.PNG)

This code was generated in the 3rd Python code block. It was calculated with the length method for all but the last method. The number of classes was calculated by turning the y_train labels into a set, and calculating the length of the set.

*Include an exploratory visualization of the dataset.*

![histogram](https://github.com/cswksu/trafficSignClassifierUdacity/blob/master/images/histograms.PNG)

3 histograms were created to gauge the distribution of traffic signs in the 3 subsets of data. The distribution across training, validation, and testing data sets seems very similar, however, the histogram isn’t flat. The first 20 signs are overrepresented compared to the last 20 signs

![sampleImg](https://github.com/cswksu/trafficSignClassifierUdacity/blob/master/images/sample.PNG)

A visualization of the first image in the training set. It is 32x32. This image will be used to illustrate the
preprocessing.

### Design and Test a Model Architecture
*Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)*


The first step of preprocessing was to pad the image with a 2 pixel border in all directions. This was done to allow the CNN to stride all the way to the edges of the image, as not all images were well centered. The 36x36 byproduct of this is shown below.

![padding](https://github.com/cswksu/trafficSignClassifierUdacity/blob/master/images/padding.PNG)

After padding, a copy of the image is created that is rotated up to 7 degrees in either direction (CW or CCW). As CNNs are not rotationally invariant, this provides additional data and helps generalize the model. A rotated image is shown below.

![tilted image](https://github.com/cswksu/trafficSignClassifierUdacity/blob/master/images/tilt.PNG)

Next, the images are converted to HSV. I’ve performed color segmentation in HSL and HSV spaces before, so I attempted to work in this color space for this project, but it may not be necessary. Finally, the image is normalized as to center the hue, saturation, and value at 0 with maximum magnitude of 1. This allows for faster convergence of weights.

*Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.*

![model scheme](https://github.com/cswksu/trafficSignClassifierUdacity/blob/master/images/arch.PNG)

The model consisted of 3 convolutional layers, which were the pooled and flattened. 4 fully connected layers then are added. The graphic above details the architecture further.

*Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.*

The model was trained over 15 epochs with batch size of 32. The loss function used was cross entropy, and the Adam Optimizer was used. Learning rate was 0.001. Dropout was utilized in the training operation. As seen in the diagram above, there are 3 opportunities for dropout in the model, all after the convolutional layers. The training set used a keep rate of 65%.


*Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.*

The model was trained for the previously stated number of epochs. If the model correctly classified 30 more validation images better than the previous best model, than it was saved. The final model achieved the following accuracies in Epoch 12:

Training: 0.997
Validation: 0.959
Test: 0.937

This downward slope from training to test implies some overfitting despite the use of dropouts. Validation accuracy can be found in the training output, with training accuracy found shortly thereafter. Test accuracy follows training accuracy.

*If an iterative approach was chosen:*
* *What was the first architecture that was tried and why was it chosen?*

An iterative solution was chosen for this project. The initial architecture was somewhat similar to LeNet 5, due to the familiarity from past lectures and projects.

* *What were some problems with the initial architecture?*

LeNet 5 is set up for greyscale images, and a simpler problem with less labels and simpler features. Number recognition is in some ways encompassed by this project in speed limit sign recognition. Additionally, initial accuracy was not great, under 90%.

* *How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.*

An additional convolutional layer was added to LeNet, and some of the pooling operations were removed. Additional fully connected layers were also added. These were added due to underfitting of the data. Additionally, more depth was added to the layers. Finally, dropout was added after the convolution layers to update LeNet to more modern practices.

* *Which parameters were tuned? How were they adjusted and why?

Batch size and epoch size were decreased and increased in order to take advantage of a GPU with large amount of fast memory and relatively fast speed. It was found that this helped in reducing underfitting. Learning rate was kept at 0.001 the whole time.

* *What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Color was chosen to be included in order to be able to distinguish between signs that are red and blue, something that a greyscale image may have a harder time conveying. Convolution layers were chosen because the sign (or more primitive features) may appear in multiple places within the bounding box. Finally, by augmenting the data, the model was more generalized than with the original training set.

### Test a Model on New Images
*Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

![web images](https://github.com/cswksu/trafficSignClassifierUdacity/blob/master/images/webRef.PNG)


The signs are all tilted slightly off their axis, leading to difficulty in classification due to the non-rotationally invariant nature of CNNs. Additionally, in downscaling the pictures to 32x32 pixels, some artifacts were created, including Moiré patterns. Finally, the stop sign appears to be shot at an angle, leading to an unwanted perspective transformation. These pictures look to be overall more vibrant than most pictures in the data set. Finally, though it is hard to see in the smaller pictures, there is a sticker attached to the face of the “turn right” sign. This noise may not be present in the original data set.

*Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).*

The model predicted all 5 images correctly, as seen below:

![predictions](https://github.com/cswksu/trafficSignClassifierUdacity/blob/master/images/pred.PNG)

These match the labels given above. Given that the test set was about 94% accurate, it stands to reason that the web images should yield an accuracy around 80-100%, assuming the model was trained with generalized data. The 100% accuracy seems reasonable enough.

*Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.*

![certainty](https://github.com/cswksu/trafficSignClassifierUdacity/blob/master/images/certainty.PNG)

The softmax probabilities for each sign are all above 94%, with most being about 1. This implies high certainty for each sign.

## Additional Discussion
This project could be further iterated on by further tweaking hyperparameters and optimizing the architecture. I have a feeling that not all of the CNN depth that I have employed is needed for good performance. Additionally, more data augmentation could be performed, using perspective transforms, Guassian blur, and other techniques.
