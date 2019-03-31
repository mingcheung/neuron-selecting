import tensorflow as tf
from cleverhans.utils_tf import train, model_eval
from cleverhans.loss import LossCrossEntropy
from ns_cifar10.basic_models import cifar10_cnn_model
from ns_cifar10.utils import data_cifar10


def train_cifar10_cnn(datadir, train_start, train_end, test_start, test_end,
                      num_epochs, batch_size, learning_rate):

    X_train, Y_train, X_test, Y_test = data_cifar10(datadir=datadir,
                                                    train_start=train_start, train_end=train_end,
                                                    test_start=test_start, test_end=test_end)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    # Training and evaluating params.
    train_params = {"nb_epochs": num_epochs, "batch_size": batch_size, "learning_rate": learning_rate}
    eval_params = {"batch_size": batch_size}

    # Define the model.
    Model = cifar10_cnn_model(input_shape=(None,) + X_train.shape[1:])
    loss = LossCrossEntropy(Model, smoothing=0.1)
    saver = tf.train.Saver(max_to_keep=1)

    x = tf.placeholder(tf.float32, shape=(None,) + X_train.shape[1:])
    y = tf.placeholder(tf.float32, shape=(None,) + Y_train.shape[1:])
    preds_x = Model.get_probs(x)


    train(sess, loss, x, y, X_train, Y_train, args=train_params)
    saver.save(sess, "./runs/ckpt/cifar10_cnn_attacked.ckpt")

    test_accuracy = model_eval(sess, x, y, preds_x, X_test, Y_test, args=eval_params)
    print("Test accuracy: %0.4f" % test_accuracy)

    sess.close()


def main(argv=None):
    train_cifar10_cnn(datadir="./data/cifar10/",
                      train_start=0, train_end=50000, test_start=0, test_end=10000,
                      num_epochs=50, batch_size=128, learning_rate=0.001)


if __name__ == "__main__":
    tf.app.run()
