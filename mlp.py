import tensorflow as tf
import numpy as np
from tensorflow.keras import utils
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import time 

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.05))

#2 layer model
def model(X, w_h1, w_h2, w_o):
    h1 = tf.nn.leaky_relu(tf.matmul(X, w_h1))
    h2 = tf.nn.leaky_relu(tf.matmul(h1, w_h2))
    return tf.matmul(h2, w_o)

#load data and split into sets, one-hot encode labels
print("\Loading LFW dataset")
lfw_people = fetch_lfw_people(data_home='.cache', min_faces_per_person=70, slice_ = (slice(75,200),slice(75,200)), resize=0.4, color=False)
_, h, w = lfw_people.images.shape 
data = lfw_people.data
target = lfw_people.target
target_labels = lfw_people.target_names
labels = utils.to_categorical(target)
trX, teX, trY, teY = train_test_split(data, labels, test_size=0.25, random_state=42)

#regularize raw data
trX = trX / 255.0 
teX = teX / 255.0

#variables used for tensorflow
X = tf.placeholder("float", [None, 2500])
Y = tf.placeholder("float", [None, 7])

learning_rate = 0.0001

#init weights
hidden_layer1 = 1000 
hidden_layer2 = 200  
size_h1 = tf.constant(hidden_layer1, dtype=tf.int32)
size_h2 = tf.constant(hidden_layer2, dtype=tf.int32)
w_h1 = init_weights([2500, hidden_layer1])
w_h2 = init_weights([hidden_layer1, hidden_layer2])
w_o = init_weights([hidden_layer2, 7])

py_x = model(X, w_h1, w_h2, w_o)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.AdamOptimizer().minimize(cost)
pred_op = tf.argmax(py_x, 1)

init = tf.global_variables_initializer()
accuracy_list = []
t1_start = time.process_time()

with tf.Session() as sess:
    print("\nMLP LFW Classification \n")
    sess.run(init)

    for k in range(100):
        for start, end in zip(range(0, len(trX), 32), range(32, len(trX) + 1, 32)): 
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        accuracy = np.mean(np.argmax(teY, axis=1) == sess.run(pred_op, feed_dict={X: teX}))

        #uncomment to see predictions
        '''if k > 90:
             pred = sess.run(pred_op, feed_dict={X:teX[k].reshape(1,2500)})
             f, (ax1) = plt.subplots(1,1)
             ax1.imshow(teX[k].reshape(50,50), interpolation="nearest", cmap='gray')
             plt.show()
             print("pred:",target_labels[pred])
             print("actual:",target_labels[np.argmax(teY[k])])
        '''
        if k % 10 == 0:
            print("Epoch:",k+1,"- Accuracy:",accuracy)
        accuracy_list.append(accuracy)
    print("Mean accuracy:", np.mean(accuracy_list))
    t1_stop = time.process_time()
    print("Time elapsed:", (t1_stop-t1_start))

    #plot accuracy
    x = np.arange(1,101,1)
    accuracy_list = [i * 100 for i in accuracy_list]
    fig = plt.figure()
    plt.plot(x,accuracy_list,marker='.')
    plt.hlines(np.mean(accuracy_list), 0, 99, label="mean:"+str(int(np.mean(accuracy_list))), colors='r')
    plt.xlabel("# of epochs")
    plt.ylabel("Accuracy")
    plt.title("TensorFlow MLP 2 layer, raw data used")
    plt.legend(loc='best')
    plt.show()

