# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
> [name=HongMing Chen] 34799
* The size of the validation set is ?
> [name=HongMing Chen] 4410
* The size of test set is ?
> [name=HongMing Chen] 12630
* The shape of a traffic sign image is ?
> [name=HongMing Chen] (32, 32, 1)
* The number of unique classes/labels in the data set is ?
> [name=HongMing Chen] 43 different labels

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. There is random image list sort by type and the bar chart showing how the data distribution.

![Fig. 1](https://i.imgur.com/F7VUKlL.png)
![Fig. 2](https://i.imgur.com/NVVlJzF.png)


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step, I decided to convert the images to grayscale. In our cases, I want to train CNN only has the data to learn geometry of signs. As a result, I remove decisive factor which is color for recognizing an traffic sign. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text](https://i.imgur.com/4ARIU06.png)

At end of preprocessing step, I normalized the image data because the piexl distribution for some objects is not uniform. This results in the CNN learning sensitive filters. 

So we training the filters with normalized distribution with which it can recognize the object with least amount of error.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| Leaky RELU			| α = 0.25												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x64 	|
| Leaky RELU			| α = 0.25											|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 			    	|
| Flatten    	      	| 5x5x64 inputs,  outputs 1600 			    	|
| Fully connected	    | 1600 inputs, outputs 400         				|
| Fully connected		| 400 inputs, outputs 120      					|
| Fully connected		| 120 inputs, outputs 43        				|

 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

> [name=HongMing Chen]To train the model, I used an ADAM optimizer for better result. I train with batch size 128 and 30 epochs. As for hyperparameter, I use learning rate 0.001 and negative slope **α** 0.25 for leaky RELU.

**Batch size**
|                   |     64   | 128      | 256      |
| :--------:          | :--------: | :--------: | :--------: |
| Test Accuracy     | 0.941    | 0.944    | 0.940    |
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Validation set accuracy = 96.2%
* Test set accuracy = 93.6%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
> [name=HongMing Chen]I start with LeNet architecture to train my classifier. I choose LeNet is because of it's outstanding performance in MNIST dataset. Also, MNIST dataset has simliar input images feature (e.g. geometry feature) to traffic sign.
* What were some problems with the initial architecture?
> [name=HongMing Chen]At first, out initial architecture performance is not good. It is because the number of hidden layer neurons isn't enough which cause the features of traffic sign are not sufficient to deal with so many sign labels.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
> [name=HongMing Chen] After trying different size of hidden layers, I change my activation function to tanh and leaky RELU. Following table shows that leaky RELU has batter accuracy imporvement and tanh remain the same as RELU sometimes even worse. As ReLU output zero for any input, this mean it won't effect final classification. Leaky ReLUs are one attempt to fix the “dying ReLU” problem. Leaky ReLU with small gradient for negative inputs give a chance to recover weight during gradient descent. Then I started to tune leaky RELU negative slope. After several trial, I set slope to 0.25. 

**Leaky relu**

|                    | Relu    | leaky Rely (alpha=0.01) | leaky Rely (alpha=0.05)  | leaky Rely (alpha=0.1) | leaky Rely (alpha=0.2) |  leaky Rely (alpha=0.25) |
| --------          | -------- | -------- | -------- | -------- | -------- | -------- |
| Test Accuracy     | 0.937    | 0.926    | 0.940    | 0.942    |  0.945   |  0.952   |

---
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
> [name=HongMing Chen] The convolution layer is aim to learn visual feature of image. The CONV layer’s parameters consist of a set of learnable filters. The network will learn filters that activate when they see some type of visual feature such as an edge of some orientation. Eventually, it will see entire honeycomb or wheel-like patterns on higher layers of the network to construct the shape of the traffic sign. 
> Besides, I use L2 regularization to prevent my model overfitting. During training phase, I add all the weight to loss function and try to mininmize the loss. In other words, it penalize the squared magnitude of all parameters directly in the objective.
> 
> I test dropout with my model. However; it can't really improve accuracy. I think dropout should work better for the network that already overfit. In turn, that means that my input data is only 32x32x1. After convolution and max-pooling, the features of image is relative less than big image. It might cause the dropout isn't perform well on this project. 
---
**Dropout**

|  | Standard | Wider features (*3) |
| --------          | -------- | -------- |
| Test Accuracy     | 0.849     | 0.845     |

W/O Dropout
|  | Standard | Wider features(*3))  |
| :--------:         | :--------: | :--------: |
| Test Accuracy     | 0.935     | 0.930     |

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

> [name=HongMing Chen]Here are five German traffic signs that I found on the web:

![](https://i.imgur.com/ZOmtDGZ.png)


> The fifth image might be difficult to classify because it has a "t" with number.
> The sixth image might be difficult to classify because it is distorted.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

> [name=HongMing Chen]Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)  | Speed limit (20km/h)   						| 
| Priority road    	    | Priority road 			        			|
| Ahead only			| Ahead only	    							|
| No entry	      		| No entry			    		 				|
| Speed limit (30km/h)	| Speed limit (30km/h)      					|
| Yield		        	| Yield      							        |
| Stop			        | Stop      							        |

> The model was able to correctly guess signs with different angle. For speed limit sign (5th in table) which has different notation ,but my model can correctly guess. That show the flexibility and generalizability.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

> [name=HongMing Chen]The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.
>
> For the first column images are testing sign which I collect them on google streetview, I print it out top five soft max probabilities and corresponding image from validtion set.

![](https://i.imgur.com/sbO0PBi.png)



