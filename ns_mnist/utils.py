import os
import shutil
import numpy as np
import pandas as pd
from pandas import DataFrame

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


def cal_keep_probs_max(file_dir, num_classes):
    keep_probs_class_all = []
    for c in range(num_classes):
        acts = np.loadtxt(os.path.join(file_dir, "activations_mean_class_" + str(c) + ".txt"))
        length = len(acts)
        keep_probs = np.zeros(length)
        keep_probs[np.argmax(acts)] = 1
        np.savetxt(os.path.join(file_dir, "keep_probs_class_" + str(c) + ".txt"),
                   keep_probs, fmt="%.f")
        keep_probs_class_all.append(keep_probs)
    np.savetxt(os.path.join(file_dir, "keep_probs_class_all.txt"),
               np.array(keep_probs_class_all), fmt="%.f")


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