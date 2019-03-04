import tensorflow as tf
import numpy as np
from tensorflow.keras import utils
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import time

batch_size = 32
test_size = 64

#function to fill 7x5 plot to show original images + top 9 patches, index parameter represents row in plot
def plotImagePatches(ax, index, image, patches):
    num_patches = patches.shape[3]
    plot_list = []
    for i in range(num_patches):
        plot_list.append((np.linalg.norm(patches[0,:,:,i]), patches[0,:,:,i]))
    plot_list.sort(key=lambda tup: tup[0], reverse=True)
    fig.suptitle("Top 4 patches - RGB input")
    ax[index][0].imshow(image, interpolation="nearest")
    ax[index][0].set_xticks([])
    ax[index][0].set_yticks([])
    for i in range(4):
        if index == 0:
            ax[index][i+1].set_title(str(i+1))
        ax[index][i+1].set_xticks([])
        ax[index][i+1].set_yticks([])
        ax[index][i+1].imshow(plot_list[i][1], interpolation="nearest")
    for i in range(7):
        ax[i][0].set_ylabel(target_names[i], rotation='horizontal', ha='right')
    
# collect first 7 samples of each class from set to plot patches
def collectExamples(images, labels):
    image_list = []
    index = 0
    tracker = 0
    while(index < len(labels)):
        if(np.argmax(teY[index]) == tracker):
            image_list.append(teX[index])
            if(len(image_list) == 10):
                break
            tracker+=1
            index = -1
        index += 1
    return image_list

def init_weights(shape, name_var):
    return tf.Variable(tf.random_normal(shape, stddev=0.05), name=name_var)

#1 layer cnn using dropout
def model_a(c1, w_fc, w_o, p_keep_conv, p_keep_hidden):
    p1 = tf.nn.max_pool(c1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME') 
    d1 = tf.nn.dropout(p1, p_keep_conv)
    rs = tf.reshape(d1, [-1, w_fc.get_shape().as_list()[0]])
    fc1 = tf.nn.relu(tf.matmul(rs, w_fc))
    d2 = tf.nn.dropout(fc1, p_keep_hidden)
    pyx = tf.matmul(d2, w_o)
    return pyx

#load and split dataset, one-hot encode labels
print("\nLoading LFW dataset")
lfw_people = fetch_lfw_people(data_home='.cache', min_faces_per_person=70, slice_ = (slice(75,200),slice(75,200)), resize=0.4, color = True)
images = lfw_people.images #50x50x3
target = lfw_people.target
target_names = lfw_people.target_names
labels = utils.to_categorical(target)
trX, teX, trY, teY = train_test_split(images, labels, test_size=0.25, random_state=42)

#regularize
trX = trX / 255.0 
teX = teX / 255.0

#variables used for tensorflow
X = tf.placeholder("float", [None, 50, 50, 3], name="X")
Y = tf.placeholder("float", [None, 7], name="Y")

#init weights
w = init_weights([3, 3, 3, 50], "w")       
w_fc = init_weights([25*25*50, 1024], "w_fc") 
w_o = init_weights([1024, 7], "w_o")         

#placeholders for dropout prob
p_keep_conv = tf.placeholder("float", name="P_K_CONV")
p_keep_hidden = tf.placeholder("float", name="P_K_HIDDEN")

#conv layer is kept outside model for patch visualisation 
conv1 = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'), name="conv1")
py_x = model_a(conv1, w_fc, w_o, p_keep_conv, p_keep_hidden)
    
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.AdamOptimizer().minimize(cost) 
predict_op = tf.argmax(py_x, 1)

init = tf.global_variables_initializer()
accuracy_list = []
t1_start = time.process_time()

with tf.Session() as sess:
    print("\nCNN LFW Classification - RGB Input\n")
    sess.run(init)

    for i in range(20): #epochs
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        num_batches = 0
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_conv: 0.8, p_keep_hidden: 0.5})
            num_batches += 1

        test_indices = np.arange(len(teX))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        
        print("Accuracy at epoch: ", end="")
        print(i+1,"=",np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices], p_keep_conv: 1.0, p_keep_hidden: 1.0})))
        accuracy_list.append(np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices], p_keep_conv: 1.0, p_keep_hidden: 1.0})))
        
    #plot top 4 patches with original image
    fig, ax = plt.subplots(7, 5)
    imgs = collectExamples(teX,teY)
    for i in range(len(imgs)):
        patches = sess.run(conv1,feed_dict={X:imgs[i].reshape(1,50,50,3), p_keep_conv: 1.0, p_keep_hidden: 1.0})
        plotImagePatches(ax, i, imgs[i], patches)
    #removed plot accuracy section, example in /figure folder
    print("Overall accuracy:", np.mean(accuracy_list))
    t1_stop = time.process_time()
    print("Time elapsed:", (t1_stop-t1_start))
    plt.show()