import numpy as np
import tensorflow as tf
from keras.utils import to_categorical

from cleverhans.utils_tf import model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks import DeepFool
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import MadryEtAl
from cleverhans.attacks import MomentumIterativeMethod

from ns_cifar10.basic_models import cifar10_cnn_model
from ns_cifar10.utils import data_cifar10
import os


def attack(datadir, train_start, train_end, test_start, test_end,
           batch_size, AttackMethod, attack_params, is_targeted, savedir):

    X_train, Y_train, X_test, Y_test = data_cifar10(datadir=datadir,
                                                    train_start=train_start, train_end=train_end,
                                                    test_start=test_start, test_end=test_end)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    # Evaluating params.
    eval_params = {"batch_size": batch_size}

    # Define the model.
    Model = cifar10_cnn_model(input_shape=(None,) + X_train.shape[1:])
    saver = tf.train.Saver(max_to_keep=1)

    AttackMethodModel = AttackMethod(Model, sess=sess)

    x = tf.placeholder(tf.float32, shape=(None,) + X_train.shape[1:])
    y = tf.placeholder(tf.float32, shape=(None,) + Y_train.shape[1:])
    preds_x = Model.get_probs(x)


    # Restore the model.
    model_file = tf.train.latest_checkpoint("./runs/ckpt/")
    saver.restore(sess, model_file)

    # Generate adversarial examples.
    # with tf.device("/cpu:0"):
    num_batches = int(X_test.shape[0] / batch_size)
    adv_X_test = []
    for i in range(num_batches):
        print("Generating adversarial examples for batch %d" % i)
        X_test_batch = X_test[i*batch_size: (i+1)*batch_size]
        Y_test_batch = Y_test[i*batch_size: (i+1)*batch_size]
        if is_targeted:
            labels = np.argmax(Y_test_batch, axis=1)
            targets = (labels+1) % Y_test_batch.shape[-1]
            Y_targets = to_categorical(targets, Y_test_batch.shape[-1])
            attack_params["y_target"] = Y_targets
        adv_X_test_batch = AttackMethodModel.generate_np(X_test_batch, **attack_params)
        adv_X_test.append(adv_X_test_batch)

    adv_X_test = np.vstack(adv_X_test)
    Y_test = Y_test[0: num_batches*batch_size]
    adversarial_accuracy = model_eval(sess, x, y, preds_x, adv_X_test, Y_test, args=eval_params)
    print("Test accuracy on adversarial examples: %0.4f" % adversarial_accuracy)


    if not os.path.exists(savedir): os.mkdir(savedir)
    if is_targeted:
        adv_file = "adversarial-examples-targeted.npy"
    else:
        adv_file = "adversarial-examples.npy"

    np.save(os.path.join(savedir, adv_file), adv_X_test)
    np.save(os.path.join(savedir, "true-labels.npy"), Y_test)

    sess.close()


def main(argv=None):
    # FastGradientMethod
    fgsm_params = {"eps": 0.03, "clip_min":0., "clip_max":1.}
    # BasicIterativeMethod
    bim_params = {"eps": 0.03, "eps_iter": 0.05, "nb_iter": 10, "clip_min":0., "clip_max":1.}
    # SaliencyMapMethod
    # jsma_params = {"theta": 1., "gamma": 0.1, "clip_min":0., "clip_max":1.}
    jsma_params = {"theta": 0.5, "gamma": 0.03, "clip_min": 0., "clip_max": 1.}
    # DeepFool
    deepfool_params = {"nb_candidate": 10, "overshoot": 0.02, "max_iter": 50, "clip_min":0., "clip_max":1.}
    # CarliniWagnerL2
    cwl2_params = {"confidence": 0, "batch_size": 128,
                   "learning_rate": 0.1, "binary_search_steps": 9,
                   "max_iterations": 10000, "abort_early": True, "initial_const": 1e-3,
                   "clip_min": 0., "clip_max": 1.}
    # MadryEtAl/PGD
    madry_params = {"eps": 0.03, "eps_iter": 0.01, "nb_iter": 40,
                    "clip_min": 0., "clip_max": 1.}
    # MomentumIterativeMethod
    mim_params = {"eps": 0.03, "eps_iter": 0.06, "nb_iter": 10, "decay_factor": 1.0,
                  "clip_min": 0., "clip_max": 1.}

    attack(datadir="./data/cifar10/",
           train_start=0, train_end=50000, test_start=0, test_end=9984,
           batch_size=64,
           AttackMethod=SaliencyMapMethod,
           attack_params=jsma_params,
           is_targeted=True,
           savedir="./data/cifar10/JSMA")


if __name__ == "__main__":
    tf.app.run()
