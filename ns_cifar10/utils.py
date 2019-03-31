import numpy as np
import pandas as pd
import pickle
import os
from pandas import DataFrame


def data_cifar10(datadir='', train_start = 0, train_end = 50000, test_start = 0, test_end = 10000):
    batch_1 = unpickle(os.path.join(datadir, 'data_batch_1'))
    batch_2 = unpickle(os.path.join(datadir, 'data_batch_2'))
    batch_3 = unpickle(os.path.join(datadir, 'data_batch_3'))
    batch_4 = unpickle(os.path.join(datadir, 'data_batch_4'))
    batch_5 = unpickle(os.path.join(datadir, 'data_batch_5'))
    test_batch = unpickle(os.path.join(datadir, 'test_batch'))

    X_train = np.vstack((batch_1[b'data'], batch_2[b'data'], batch_3[b'data'], batch_4[b'data'], batch_5[b'data']))
    X_train = np.reshape(X_train, (-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    Y_train = np.array([batch_1[b'labels'], batch_2[b'labels'], batch_3[b'labels'], batch_4[b'labels'], batch_5[b'labels']])
    Y_train = np.reshape(Y_train, (-1, 1))
    Y_train = one_hot_encoding(Y_train, 10)

    X_test = test_batch[b'data']
    X_test = np.reshape(X_test, (-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    Y_test = np.array(test_batch[b'labels'])
    Y_test = np.reshape(Y_test, (-1, 1))
    Y_test = one_hot_encoding(Y_test, 10)

    X_train = X_train[train_start:train_end]
    Y_train = Y_train[train_start:train_end]
    X_test = X_test[test_start:test_end]
    Y_test = Y_test[test_start:test_end]

    # Normalize to 0-1.
    X_train = X_train / 255.
    X_test = X_test / 255.

    return X_train, Y_train, X_test, Y_test


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
    return dict


def one_hot_encoding(labels, num_classes):
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    one_hot_labels = np.zeros((num_labels, num_classes), dtype = np.float32)
    one_hot_labels.flat[index_offset + labels.ravel()] = 1.0
    return one_hot_labels


def analyze_training_activations(file_dir, start_batch, end_batch, num_classes):
    # Read activations and labels to data frames.
    all_df = DataFrame()
    for i in range(start_batch, end_batch):
        activations_file = os.path.join(file_dir, "activations_batch_" + str(i) + ".txt")
        labels_file = os.path.join(file_dir, "labels_batch_" + str(i) + ".txt")
        activations_df = pd.read_csv(activations_file, sep=" ", header=None)
        labels_df = pd.read_csv(labels_file, header=None, names=["class"])
        all_df = all_df.append(pd.concat([activations_df, labels_df], axis=1))
    # Filter data according to different class.
    for c in range(num_classes):
        all_df_c = all_df[all_df["class"] == c]
        # all_df_c_sum = all_df_c.apply(lambda x: x.sum(), axis=0)
        all_df_c_mean = all_df_c.apply(lambda x: x.mean(), axis=0)
        del all_df_c["class"]
        all_df_c.to_csv(os.path.join(file_dir, "activations_class_" + str(c) + ".txt"),
                        float_format="%.3f", index=False, header=False)
        # all_df_c_sum.drop(["class"], inplace=True)
        # all_df_c_sum.to_csv(os.path.join(file_dir, "activations_sum_class_" + str(c) + ".txt"),
        #                     float_format="%.3f", index=False, header=False)
        all_df_c_mean.drop(["class"], inplace=True)
        all_df_c_mean.to_csv(os.path.join(file_dir, "activations_mean_class_" + str(c) + ".txt"),
                            float_format="%.3f", index=False, header=False)


def cal_keep_probs_cdf(file_dir, p, num_classes):
    keep_probs_class_all = []
    for c in range(num_classes):
        acts = np.loadtxt(os.path.join(file_dir, "activations_mean_class_" + str(c) + ".txt"))
        acts_freq = acts / np.sum(acts)
        length = len(acts_freq)
        if p == 1.0:
            keep_probs = np.ones(length)
        else:
            acts_freq_sort = -np.sort(-acts_freq)
            acts_freq_argsort = np.argsort(-acts_freq)
            cdf = np.add.accumulate(acts_freq_sort)
            keep_num = np.sum(cdf <= p)
            keep_probs = np.zeros(length)
            keep_probs[acts_freq_argsort[0: keep_num]] = 1
        np.savetxt(os.path.join(file_dir, "keep_probs_class_" + str(c) + ".txt"),
                   keep_probs, fmt="%.f")
        keep_probs_class_all.append(keep_probs)
    np.savetxt(os.path.join(file_dir, "keep_probs_class_all.txt"),
               np.array(keep_probs_class_all), fmt="%.f")


def get_keep_probs(selected_layers, is_selected, num_classes, num_neurons):
    keep_probs = []
    for layer in selected_layers:
        if is_selected[layer]:
            keep_prob_layer = np.loadtxt("./runs/train/" + layer + "/keep_probs_class_all.txt")
            keep_probs.append(np.array(keep_prob_layer))
        else:
            keep_prob_layer = np.ones((num_classes, num_neurons[layer]))
            keep_probs.append(keep_prob_layer)
    return keep_probs
