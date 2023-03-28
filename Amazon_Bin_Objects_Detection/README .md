# Data Preprocessing

## Data Engineering
Before model training, images were normalized:

1) Re-sized to 224x224 pixels
2) Tansformed for zero mean, and unit variance
3) For convolutional models, the dataset was augmented with horizontal flips of



## Feature Engineering
### Applying Edge detection and Histogram of Oriented Gradients

the tape wrapping the products in the bin skews features so we tried to remove it by subtract the fore ground which contain the tape and extract the background contain the products this process had been done by using canny edge detection to detect the edges between the mean and the maximum of the image as a mask to extract the fore ground then subtracting it from the image and doing the same thing by using the Histogram of Oriented Gradients

### Blobs
We explored the Blob features extraction. Blobs are bright on dark or dark on bright regions in an image.
We used Laplacian of Gaussian approach, it computes the Laplacian of images with successively increasing standard deviation and stacks them up in a cube. Blobs are local maximas in this cube.

# Model Building

## 1. Using Support Vector Machine
In order to determine which kernel had the maximum accuracy for the SVC, we chose to test three different kernels while using HOG and PCA feature extraction.
The kernels that were used are:
  1. Linear SVC
  2. Poly SVC
  3. RBF SCV

### SVC with PCA feature extraction:
Principal Component Analysis (PCA) is a common feature extraction method in data science. Technically, PCA finds the eigenvectors of a covariance matrix with the highest eigenvalues and then uses those to project the data into a new subspace of equal or less dimensions. Practically, PCA converts a matrix of n features into a new dataset of (hopefully) less than n features. That is, it reduces the number of features by constructing a new, smaller number of variables which capture a significant portion of the information found in the original features. In this part of our project we paired this PCA feature extraction with the three types of kernels that were mentioned above.

#### Linear Kernel:
  A Linear Support Vector Classifier’s goal is to divide or classify the data we supply by returning a "best fit" hyperplane. We may then feed some features to our classifier to get the "predicted" class after acquiring the hyperplane.

  However, because linear SVC didn't go well with the kind of data we were using, when we applied it with PCA feature extraction, we only received 0.1343 which is really unreliable and not usable.

#### Poly Kernel:
  A Polynomial Support Vector Classifier’s goal is supposedly a more accurate way to divide or classify the data we supply by returning a "best fit" hyperplane in order to then feed some features to our classifier to get the "predicted" class after acquiring the hyperplane.

  By applying Poly SVC with PCA feature extraction, the accuracy only reached 0.1694 which is a tiny bit better than the linear SVC but still is unreliable.

#### RBF kernel:
  A Support Vector Machine with RBF Kernel is a machine learning algorithm which is capable of classifying data points separated with radial based shapes. Therefore, we tried this kernel out in hopes of it giving better results with our dataset but it gave us an accuracy of 0.1694 which is the same as Poly Kernel with PCA.

### SVC with HOG feature extraction:
Histogram of Gradient (HOG) is a technique which counts occurrences of gradient orientation in localized portions of an image. This method is similar to that of edge of histogram, scale-invariant feature transform descriptors, and shape contexts, but differs in that it is computed on a dense grid of uniformly spaced cells and uses overlapping local contrast normalization for improved accuracy.

#### Linear kernel:
  When we applied a linear kernel with HOG feature extraction the best accuracy that we reached was 0.008 which was unusable.

#### Poly kernel:
  By applying Poly SVC with HOG feature extraction, the accuracy only reached 0.1732 which is a little bit better than the linear SVC but still is unreliable.

#### RBF kernel:
  We tried this kernel out in order to fit better with our dataset to give us a better accuracy but it gave us an accuracy of 0.1732 which is the same as Poly Kernel with PCA.

### SVC with LoG feature extraction:
Laplacian of Gaussian (LoG) is used for blobs detection and it is the most accurate and slowest approach. It computes the Laplacian of Gaussian images with successively increasing standard deviation and stacks them up in a cube. Blobs are local maximas in this cube. Detecting larger blobs is especially slower because of larger kernel sizes during convolution. Only bright blobs on dark backgrounds are detected. 

#### Linear kernel:
We reached an accuracy of 0.129 with the Linear kernel on LOG feature extraction which was unreliable.

#### Poly kernel:
By applying Poly kernel on LOG feature extraction the accuracy reached 0.128 which can't be used on our dataset.

#### RBF kernel:
Lastly, RBF kernel was applied and the accuracy ws 0.17 which was better than the previous kernels but it was still unusable.

### Conclusion on SVC:

We decided to work on this brute force solution in order to find the best model for our dataset. However, we learned that SVM is not the best solution and that's why we decided to work on Convoluation Neural Network (CNN) next.

## 2. Using Convoluation Neural Network
1. At first Model we have a x_train and making a hog for all image have 1 d array 
    so we use Conv1D and have filter = 64 and input shape=X_train[1]
	and droupout=0.5
	and kernal size =3
	and make one layer of Max pooling (pooling_size=2)
	added flaten layer ,used for out put softmax layer with 2 classes ,loss=categorical_crossentropy,optemizer =Adam
	acc=17.5
2. updated model 
     we increase data ,still making a hog for all image
	 and make filters =128 , and make 2 layer of max pooling between each layer of Conv1D
	 remove droupout layer 
	 used for out put sigmoid instead of softmax  of 6 classes
	 acc=25.6
3.  second updated 
      we work on raw data and make balance for data 
	  make same layer of last version of model 
	  acc=30.6
## 3. Using CNN Pre-trained Models
we used different pre-trained models for feature extractions from our images data by freezing all the weights of the neural network and only changes the classification
layer so the model only need to train these weights at the last classification layer, we used fixed learning rate by 0.001 and the loss function was categorical cross entropy for 5 classes
- ResNet34 with SGD as an Optimizer with 0.9 momentum: the model overfitted very fast and the validation accuracy was 44.9%
- ResNet50 with SGD as an Optimizer with 0.9 momentum: The model achieved 43.7 validation accuracy, we decided to use Adam as our optimizer
- VGG16: The model was very weak although we trained over 30 epochs but the train and the validation accuracy didn't exceed 50% so the approach wasn't good choice for the features extraction
- ResNet50: this model results was promising actually as the model achieved 50% validation accuracy and 48% test accuracy
- ResNet34: This was the best validation and test accuracy the validation reaches 55% and **the final test accuracy was 50.5%** as we noticed the data was very noisy and most of the images hard to be determined number of items inside the image so this was accepted results for our project
# Model Deployment

 ![img_4.png](img_4.png)
Now we are done with the model part and got the accuracy of model,
now its turn to deploy our model.

For Deployment of model we used ngrok.
It  is the programmable network edge that adds connectivity,
security, and observability to your apps.

**Steps:**

1) Load the pytorch model.

 ![img_2.png](img_2.png)
 

2) Import necessary  libraries.
 
![img_3.png](img_3.png)



3) Connect with host server and deploy the pytorch model on ngrok.


**Workflow**- 

1) Once model is deployed the first view of application will be -

![img_1.png](img_1.png)


2) Uploading the Amazon bin image data



![img.png](img.png)

3) Once we click on predict button, the result will be shown as object count in the inventory box. 
The output of our application will be number of object in the box. 


(Note : Image need to be attached here once server is active)



