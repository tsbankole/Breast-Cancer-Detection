# Breast-Cancer-Detection

## Definition
### Overview
This project is based on the application of artificial intelligence to breast cancer detection. Digitized images of tissues have been analyzed for feature extraction in the open literature [1]. The diagnosis of a tissue is either malignant (invasive and dangerous) or benign (non-invasive). The identification and classification of a cancerous tumor can greatly improve the chances of a patient surviving treatment as early diagnosis and treatment are very pertinent to beating cancer. In addition to the benefits of early detection, using machine learning and artificial intelligence for breast cancer detection is racially unbiased compared to traditional models [2]. Finally, breast cancer risk factors are up from 5.1% in 1940 to 12% now [3]. It is therefore an imperative study to conduct.

Given the features extracted from the digitized images of the tissues, the task is to detect whether the presented tumor is benign or malignant.
The solution of this problem will be approached by building an artificial neural network (ANN) in PyTorch which will produce a probability 𝑝∈[0,1] of the tissue being malignant. Here the null hypothesis is that the tissue is benign (𝑝=0 ) and the alternative hypothesis is that the tissue is malignant ( 𝑝 = 1 ).
To build this Neural Network, activation functions that work well for classification would be employed. This would include relU (Rectified Linear Unit) and tanh (Hyperbolic tangent). To prevent overfitting, the number of neurons in the network will be calculated according to this formula [4]

𝑁ℎ=𝑁𝑠/(𝛼∗(𝑁𝑖+𝑁𝑜))

Ni = number of input neurons. 

No = number of output neurons. 

Ns = number of samples in training data set. 

α = an arbitrary scaling factor usually 2-10

### Metrics
Binary Cross Entropy would be used as a loss function for training the Artificial Neural Network as the task here is a binary classification. The test accuracy (%) would be used as a comparison metric against the results in the paper [1].

Data Exploration Ten real-valued features are computed for each cell nucleus: a) radius (mean of distances from center to points on the perimeter) b) texture (standard deviation of gray-scale values) c) perimeter d) area e) smoothness (local variation in radius lengths) f) compactness (perimeter^2 / area - 1.0) g) concavity (severity of concave portions of the contour) h) concave points (number of concave portions of the contour) i) symmetry j) fractal dimension ("coastline approximation" - 1). In addition to these features, the mean, standard error and worst of these features are also computed leading to a total of 30 features. It is therefore necessary to perform dimensionality reduction to find linear combinations of features that capture variability in the data.

## Algorithm

Aritificial Neural Network in PyTorch with activation functions such as relU and tanh were used in Amazon Sagemaker to train the Neural Network. A training accuracy of 98.6% and a test accuracy of 98.3% was obtained.

## References
1. Bennett, Kristin P., and Olvi L. Mangasarian. "Robust linear programming discrimination of two linearly inseparable sets." Optimization methods and software 1, no. 1 (1992): 23-34.
2. https://giving.massgeneral.org/artificial-intelligence-breast-cancer-detection/
3. https://www.webmd.com/breast-cancer/breast-cancer-detection
4. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
