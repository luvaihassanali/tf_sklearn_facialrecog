import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras import utils
import time 

#load data, separate into training/testing sets, and one-hot encode labels
print("\nLoading LFW dataset")
lfw_people = fetch_lfw_people(data_home='.cache', min_faces_per_person=70, slice_ = (slice(75,200),slice(75,200)), resize=0.4, color=False)
_, height, width = lfw_people.images.shape 
images = lfw_people.data 
labels = lfw_people.target
labels = utils.to_categorical(labels)
target_labels = lfw_people.target_names
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, random_state=42) 

#function to plot 3x4 display of first 12 images of the parameter 
def plotQuick(images):
    for i in range(12):
        img = images[i].reshape(50,50)
        plt.subplot(3,4,1+i)
        plt.imshow(img, interpolation="nearest", cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.margins(0)

#compute average face of data to normalize training set
mean_face = np.zeros((1,height*width))
for i in X_train:
    mean_face = np.add(mean_face,i)
mean_face = np.divide(mean_face,float(len(y_train))).flatten()
normalized_X_train = np.ndarray(shape=(len(y_train), height*width))
for i in range(len(y_train)):
    normalized_X_train[i] = np.subtract(X_train[i],mean_face)

#compute pca to extract eigenfaces
pca = PCA(n_components=50, svd_solver='randomized', whiten=True).fit(X_train)
eigenfaces = pca.components_ #eigenvectors of covariance matrix

#Figure 1 
plotQuick(eigenfaces) #plots first 12 eigenfaces
plt.suptitle("First 12 Eigenfaces (PCA, n_components=50)")

#compute weights of training set 
weights = np.array([np.dot(eigenfaces,i) for i in normalized_X_train])

print("Classifying images in test set")
accuracy = 0
t1_start = time.process_time()
for i in range(len(X_test)):
    unknown_face = X_test[i]
    normalized_uface = np.subtract(unknown_face,mean_face)
    w_unknown = np.dot(eigenfaces, normalized_uface) #compute weights of unknown image using existing eigenfaces

    #distance in eigenspace is approximately equal to the correlation between two images
    norm = np.linalg.norm(weights-w_unknown,axis=1)
    index = np.argmin(norm)
    #Figure 2 test cases were plotted using the commented section below
    '''if i > 270: 
            print(target_labels[np.argmax(y_train[index])],target_labels[np.argmax(y_test[i])])
            fig, (ax1, ax2, ax3) = plt.subplots(1,3)
            fig.suptitle("Test image: "+target_labels[np.argmax(y_test[i])]+"\nPredicted image: "+target_labels[np.argmax(y_train[index])])
            ax1.imshow(unknown_face.reshape(50,50), cmap='gray', interpolation='nearest')
            ax2.imshow(normalized_uface.reshape(50,50), cmap='gray', interpolation='nearest')
            ax3.imshow(X_train[index].reshape(50,50), cmap='gray', interpolation='nearest')
            ax1.set_xlabel("Original")
            ax2.set_xlabel("Normalized")
            ax3.set_xlabel("Prediction")
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax3.set_xticks([])
            ax3.set_yticks([])
            plt.show()'''
    #if prediction is correct increase accuracy counter 
    if (target_labels[np.argmax(y_train[index])] == target_labels[np.argmax(y_test[i])]):
        accuracy += 1

print("Accuracy:",accuracy / len(X_test))
t1_stop = time.process_time()
print("Time elapsed:", (t1_stop-t1_start))
plt.show()