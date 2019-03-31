import os
import shutil
import numpy as np
import tensorflow as tf
from ns_mnist import input_data
from ns_mnist.utils import analyze_training_activations
from ns_mnist.utils import cal_keep_probs_max
from ns_mnist.utils import cal_keep_probs_cdf
from ns_mnist.utils import get_keep_probs


###====================Define parameters======================
selected_layers = ["conv1", "conv2", "pool3", "conv4", "conv5", "pool6", "fc7", "fc8"]
is_selected = {"conv1":False,    "conv2":False,    "pool3":False,
               "conv4":True,     "conv5":True,     "pool6":True,
               "fc7"  :True,     "fc8"  :True}
num_neurons = {"conv1":26*26*32, "conv2":24*24*32, "pool3":12*12*32,
               "conv4":10*10*64, "conv5":8*8*64,   "pool6":4*4*64,
               "fc7"  :200,      "fc8"  :200}
p_values    = {"conv1":1,        "conv2":1,        "pool3":1,
               "conv4":0.9,      "conv5":0.8,      "pool6":0.7,
               "fc7"  :0.95,     "fc8"  :"max"}
adv_method = "CWL2" # "normal", "FGSM", "BIM", "JSMA", "DeepFool", "CWL2", "Madry", "MIM".
is_targeted = True
batch_size = 128
num_classes = 10


###======================mnist-cnn-model======================
# Input placeholers.
x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(tf.float32, shape=[None, num_classes])
conv1_keep_probs = tf.placeholder(tf.float32, shape=[num_classes, None])
conv2_keep_probs = tf.placeholder(tf.float32, shape=[num_classes, None])
pool3_keep_probs = tf.placeholder(tf.float32, shape=[num_classes, None])
conv4_keep_probs = tf.placeholder(tf.float32, shape=[num_classes, None])
conv5_keep_probs = tf.placeholder(tf.float32, shape=[num_classes, None])
pool6_keep_probs = tf.placeholder(tf.float32, shape=[num_classes, None])
fc7_keep_probs = tf.placeholder(tf.float32, shape=[num_classes, None])
fc8_keep_probs = tf.placeholder(tf.float32, shape=[num_classes, None])
keep_probs_ph = [conv1_keep_probs, conv2_keep_probs, pool3_keep_probs,
                 conv4_keep_probs, conv5_keep_probs, pool6_keep_probs,
                 fc7_keep_probs,   fc8_keep_probs]


# Basic necessary functions.
def conv2d(x, w):
    return tf.nn.conv2d(x, filter=w, strides=[1, 1, 1, 1], padding="VALID")
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

# Neuron-selecting function.
# def neuron_select(x, keep_probs):
#     mingle = tf.reduce_sum(x * keep_probs[0]) / (tf.norm(x) * tf.norm(keep_probs[0]))
#     x_selected = x * keep_probs[0]
#     for i in range(1, keep_probs.shape[0]):
#         mingle_tmp = tf.reduce_sum(x * keep_probs[i]) / (tf.norm(x) * tf.norm(keep_probs[i]))
#         x_selected, mingle = tf.cond(mingle_tmp > mingle,
#                                      lambda: (x * keep_probs[i], mingle_tmp),
#                                      lambda: (x_selected, mingle))
#     return x_selected

# Neuron-selecting function.
def neuron_select(x, keep_probs, type, k):
    if type=="cos":
        mingle = tf.reduce_sum(x * keep_probs, axis=1) / (tf.norm(x) * tf.norm(keep_probs, axis=1))
    elif type=="max":
        mingle = tf.reduce_sum(x * keep_probs, axis=1)
    k_indices = tf.nn.top_k(mingle, k=k, sorted=True).indices
    k_keep_probs = keep_probs[k_indices[0]]
    for index in range(1, k_indices.shape[0]):
        k_keep_probs += keep_probs[k_indices[index]]
    k_keep_probs = tf.sign(k_keep_probs)
    return  x * k_keep_probs

# Selecting function wrapper.
def select_wrapper(h_acts, keep_probs, type, k):
    return tf.map_fn(fn=lambda x: neuron_select(x, keep_probs, type, k),
                     elems=h_acts, dtype=tf.float32)

# Read the trained model.
reader = tf.train.NewCheckpointReader("./runs/ckpt/mnist_cnn_attacked.ckpt")

# CONV1: Convolution + ReLU.
# input: 28x28x1, filters: 3x3x32, output: 26x26x32.
w_conv1 = reader.get_tensor("Variable")
b_conv1 = reader.get_tensor("Variable_1")
h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)

# CONV1 Selected.
h_conv1_flat = tf.reshape(h_conv1, [-1, 26*26*32])
h_conv1_flat_selected = select_wrapper(h_acts=h_conv1_flat,
                                       keep_probs=conv1_keep_probs,
                                       type="cos", k=1)
h_conv1_selected = tf.reshape(h_conv1_flat_selected, [-1, 26, 26, 32])

# CONV2: Convolution + ReLU.
# input: 26x26x32, filters: 3x3x32, output: 24x24x32.
w_conv2 = reader.get_tensor("Variable_2")
b_conv2 = reader.get_tensor("Variable_3")
h_conv2 = tf.nn.relu(conv2d(h_conv1_selected, w_conv2) + b_conv2)

# CONV2 Selected.
h_conv2_flat = tf.reshape(h_conv2, [-1, 24*24*32])
h_conv2_flat_selected = select_wrapper(h_acts=h_conv2_flat,
                                       keep_probs=conv2_keep_probs,
                                       type="cos", k=1)
h_conv2_selected = tf.reshape(h_conv2_flat_selected, [-1, 24, 24, 32])

# POOL3: Max Pooling.
# input: 24x24x32, pooling: 2x2, output: 12x12x32.
h_pool3 = max_pool_2x2(h_conv2_selected)

# POOL3 Selected.
h_pool3_flat = tf.reshape(h_pool3, [-1, 12*12*32])
h_pool3_flat_selected = select_wrapper(h_acts=h_pool3_flat,
                                       keep_probs=pool3_keep_probs,
                                       type="cos", k=1)
h_pool3_selected = tf.reshape(h_pool3_flat_selected, [-1, 12, 12, 32])

# CONV4: Convolution + ReLU.
# input: 12x12x32, filters: 3x3x64, output: 10x10x64.
w_conv4 = reader.get_tensor("Variable_4")
b_conv4 = reader.get_tensor("Variable_5")
h_conv4 = tf.nn.relu(conv2d(h_pool3_selected, w_conv4) + b_conv4)

# CONV4 Selected.
h_conv4_flat = tf.reshape(h_conv4, [-1, 10*10*64])
h_conv4_flat_selected = select_wrapper(h_acts=h_conv4_flat,
                                       keep_probs=conv4_keep_probs,
                                       type="cos", k=1)
h_conv4_selected = tf.reshape(h_conv4_flat_selected, [-1, 10, 10, 64])

# CONV5: Convolution + ReLU.
# input: 10x10x64, filters: 3x3x64, output: 8x8x64.
w_conv5 = reader.get_tensor("Variable_6")
b_conv5 = reader.get_tensor("Variable_7")
h_conv5 = tf.nn.relu(conv2d(h_conv4_selected, w_conv5) + b_conv5)

# CONV5 Selected.
h_conv5_flat = tf.reshape(h_conv5, [-1, 8*8*64])
h_conv5_flat_selected = select_wrapper(h_acts=h_conv5_flat,
                                       keep_probs=conv5_keep_probs,
                                       type="cos", k=1)
h_conv5_selected = tf.reshape(h_conv5_flat_selected, [-1, 8, 8, 64])

# POOL6: Max Pooling.
# input: 8x8x64, pool: 2x2, output: 4x4x64.
h_pool6 = max_pool_2x2(h_conv5_selected)
h_pool6_flat = tf.reshape(h_pool6, [-1, 4*4*64])

# POOL6 Selected.
h_pool6_flat_selected = select_wrapper(h_acts=h_pool6_flat,
                                       keep_probs=pool6_keep_probs,
                                       type="cos", k=1)

# FC7: Fully Connected + ReLU.
# input: 4*4*64, output: 200.
w_fc7 = reader.get_tensor("Variable_8")
b_fc7 = reader.get_tensor("Variable_9")
h_fc7 = tf.nn.relu(tf.matmul(h_pool6_flat_selected, w_fc7) + b_fc7)

# FC7 Selected.
h_fc7_selected = select_wrapper(h_acts=h_fc7,
                                keep_probs=fc7_keep_probs,
                                type="cos", k=1)

# FC8: Fully Connected + ReLU.
# input: 200, output: 200.
w_fc8 = reader.get_tensor("Variable_10")
b_fc8 = reader.get_tensor("Variable_11")
h_fc8 = tf.nn.relu(tf.matmul(h_fc7_selected, w_fc8) + b_fc8)

# Selected.
h_fc8_selected = select_wrapper(h_acts=h_fc8,
                                keep_probs=fc8_keep_probs,
                                type="cos", k=1)

# OUTPUT: Softmax.
# input: 200, output: 10.
w_out = reader.get_tensor("Variable_12")
b_out = reader.get_tensor("Variable_13")
logits_out = tf.matmul(h_fc8_selected, w_out) + b_out
y_out = tf.nn.softmax(logits_out)

# Accuracy.
correct_preds = tf.equal(tf.argmax(y_out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
###=======End==================================================


###=======Using a subset(about 100 batches) of MNIST train data
###========to save the correct activations of the layers to be selected.

# traindata = input_data.read_mnist_data_sets(data_dir="./data/mnist", one_hot=True).train
# num_batches = 100
# is_selected = {"conv1":False,    "conv2":False,    "pool3":False,
#                "conv4":False,    "conv5":False,    "pool6":False,
#                "fc7"  :False,    "fc8"  :False}
# keep_probs = dict(zip(keep_probs_ph, get_keep_probs(selected_layers, is_selected, num_classes, num_neurons)))
# sess = tf.Session()
# for b in range(num_batches):
#     print(b)
#     examples = traindata.next_batch(batch_size)
#     inputs = {x: examples[0], y: examples[1]}
#     ops = [h_conv1_flat, h_conv2_flat, h_pool3_flat,
#            h_conv4_flat, h_conv5_flat, h_pool6_flat,
#            h_fc7,        h_fc8,        correct_preds]
#     op_values = sess.run(ops, feed_dict={**inputs, **keep_probs})
#     batch_correct_preds = op_values[-1]
#     for i in range(len(selected_layers)):
#         if not os.path.exists("./runs/train/" + selected_layers[i]):
#             os.mkdir("./runs/train/" + selected_layers[i])
#         np.savetxt("./runs/train/" + selected_layers[i] + "/activations_batch_" + str(b) + ".txt",
#                    op_values[i][batch_correct_preds==True], fmt="%0.4f")
#         np.savetxt("./runs/train/" + selected_layers[i] + "/labels_batch_" + str(b) + ".txt",
#                    np.argmax(examples[1], 1)[batch_correct_preds==True], fmt="%d")

# # Analyze activations of the hidden layers to be selected.
# for layer in selected_layers:
#     analyze_training_activations(file_dir=os.path.join("./runs/train/", layer),
#                                  start_batch=0,
#                                  end_batch=num_batches,
#                                  num_classes=num_classes)
# sess.close()
###======end===================================================================



###=========Accuracy of adversarial examples after neuron-selecting.===========
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
if adv_method=="normal":
    testdata = input_data.read_mnist_data_sets(data_dir="./data/mnist", one_hot=True).test
else:
    testdata = input_data.read_adv_data_sets(advdata_dir="./data/mnist/" + adv_method,
                                             is_targeted=is_targeted).data

num_tests = testdata.images.shape[0]
num_batches = int(num_tests / batch_size)
total_correct_preds = []

for layer in selected_layers:
    if p_values[layer] == "max":
        cal_keep_probs_max(file_dir="./runs/train/" + layer, num_classes=num_classes)
    else:
        cal_keep_probs_cdf(file_dir="./runs/train/" + layer, p=p_values[layer], num_classes=num_classes)
keep_probs = dict(zip(keep_probs_ph, get_keep_probs(selected_layers, is_selected, num_classes, num_neurons)))

for b in range(num_batches):
    examples = testdata.next_batch(batch_size)
    inputs = {x:examples[0], y:examples[1]}
    batch_correct_preds = sess.run(correct_preds, feed_dict={**inputs, **keep_probs})
    total_correct_preds.append(batch_correct_preds)

total_accuracy = np.mean(np.concatenate(total_correct_preds))
print("Accuracy of test examples: %.4f" % total_accuracy)
sess.close()
#============End==================================================================
