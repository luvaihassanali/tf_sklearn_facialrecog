import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import time

#plot first 12 images of parameter along with inputted titles
def plotQuick(images, titles):
    for i in range(12):
        img = images[i].reshape(50,50)
        plt.subplot(3,4,1+i)
        plt.imshow(img, interpolation="nearest", cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title(titles[i], size=10)
        plt.margins(0)
    plt.suptitle("KNN Projections")

#functions to return titles for images being plotted - adapted from sklearn example
def name_title(predictions, i):
    pred_name = target_names[predictions[i]].rsplit(' ', 1)[-1]
    return pred_name

def prediction_title(predictions, actual, target_names, i):
    pred_name = target_names[predictions[i]].rsplit(' ', 1)[-1]
    true_name = target_names[actual[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

#load data and split into test/train sets, one-hot encode labels
print("\nLoading LFW dataset")
lfw_people = fetch_lfw_people(data_home='.cache', min_faces_per_person=70, slice_ = (slice(75,200),slice(75,200)), resize=0.4, color=False)
_, h, w = lfw_people.images.shape
images = lfw_people.data
labels = lfw_people.target
target_names = lfw_people.target_names
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, random_state=42)

#compute pca to extract eigenfaces
pca = PCA(n_components=50, svd_solver='randomized', whiten=True).fit(X_train)
eigenfaces = pca.components_.reshape((50, h, w))

#project input data into eigenspace
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

t1_start = time.process_time()

#change classifier being used by uncommenting and commenting the other
#clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=32, verbose=True)
clf = KNeighborsClassifier(n_neighbors=1) #using a K means Classifier
clf.fit(X_train_pca, y_train)

#if using mlp classifier you can uncomment this block to plot loss per epoch
'''
plt.title("Scikit-Learn MLPClassifier 'adam' Loss Function")
plt.xlabel("# of epochs")
plt.ylabel("Loss")
plt.plot(clf.loss_curve_)
plt.show()
'''
#make predictions
y_pred = clf.predict(X_test_pca)

#test accuracy of predictions made
print("Classifying images in test set")
acc = 0
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        acc += 1
print("Mean accuracy:", acc/len(y_test))
t1_stop = time.process_time()
print("Time elapsed:", (t1_stop-t1_start))

#create titles for plots
prediction_titles = [prediction_title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]
plain_titles = [name_title(y_pred, i)
                     for i in range(y_pred.shape[0])]

projections = pca.inverse_transform(X_test_pca) #load projected images
plotQuick(X_test, prediction_titles) #plot predictions
plt.show()
plotQuick(projections, plain_titles) #plot reconstructed
plt.show() #must display seperately, using same axis
