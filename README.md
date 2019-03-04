# tf_sklearn_facialrecog

Evaluation of Eigenfaces System
Luvai Hassanali   
*sklearn, tensorflow, and matplotlib required (installed using pip3)

Notes:
- /.cache folder contains LFW dataset
- /experiments folder contains four .txt files with results from experimentation 
- /figures folder contains images used in paper and additional ones which were excluded 
There are 6 models discussed in my paper and 5 .py files (#2 contains two models).
Each .py file can be run as is to display accuracy and time measurements discussed about in my paper:

1. eigenfaces.py has the basic eigenface system:
  - running the file will load LFW set, split into test/training, calculate weights of training set, then classify the test set
  - the file will also display the first 12 eigenvectors computed by PCA 
  - line 61 - 67 can be uncommented and the last 52 predictions will be visually displayed, examples of these figures are used in Figure 2 of my paper
  
2. eigenfaces_v2.py contains implementation for Scikit-Learn eigenface system AND Scikit-Learn mlp model:
  - the .py file is setup to run scikit-learn eigenface system which uses KNN algorithm 
  - it will load LFW set, split into test/training, project into eigenspace, then classify the test set
  - this file also contains implementation of mlp fully dependent on Scikit-Learn which is an experiment discussed in my paper
  - to change from knn to mlp implementation all you need to do is uncomment line 52 and comment line 53 
  - with mlp classifier epoch + loss will be printed to console during execution and you can uncomment line 58-62 to display a plot of this
  - the end of the file will display plot for predictions and projections separately, regardless of implementation

3. mlp.py contains TensorFlow implementation of MLP:
  - the file will load data and train for 100 epochs
  - accuracy is displayed to console every 10 epochs 
  - line 64 - 71 can be uncommented to visualize last 10 epoch predictions
  - plot of accuracy over 100 epochs is displayed at end

4. cnn_grey.py has implementation of CNN with greyscale input:
  - the file will load and split data then begin training for 20 epochs
  - accuracy at epoch will be printed to console for each epoch 
  - top 4 patch plots along with orignal images (one for each class) will display at end
 
5. cnn_rgb.py has implementation of CNN with RGB input:
  - the file will load and split data then begin training for 20 epochs
  - accuracy at epoch will be printed to console and patch plots will display at end 
  - you may find it useful to reduce number of epochs as each version of the cnn takes about 10 minutes for 20

The /experiments folder contains four text files each with examples of accuracy results for the network specified:
- cnn_greyscale_accuracy_different_ksize: results from testing different ksize on max pool layer on CNN with greyscale input
- cnn_rgb_accuracy_different_ksize: results from teseting different ksize on max pool layer but with CNN that takes RGB input
- mlp_accuracy_different_weights: results from different weight configurations of the 2 hidden layers of TensorFlow implementation of MLP
- sklearn_mlp_loss_different_hidden: results from mlp fully-dependent on Scikit-Learn library from testing different number of hidden layer sizes
