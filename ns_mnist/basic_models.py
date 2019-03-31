from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from cleverhans.model import Model


class BasicModel(Model):
    """An example of a bare bone of multi-layer model class."""
    def __init__(self, layers, input_shape):
        super(BasicModel, self).__init__()

        self.layer_names = []
        self.layers = layers
        self.input_shape = input_shape
        if isinstance(layers[-1], Softmax):
            layers[-1].name = 'probs'
            layers[-2].name = 'logits'
        else:
            layers[-1].name = 'logits'
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'name'):
                name = layer.name
            else:
                name = layer.__class__.__name__ + str(i)
            self.layer_names.append(name)

            layer.set_input_shape(input_shape)
            input_shape = layer.get_output_shape()

    def fprop(self, x, set_ref=False):
        states = []
        for layer in self.layers:
            if set_ref:
                layer.ref = x
            x = layer.fprop(x)
            assert x is not None
            states.append(x)
        states = dict(zip(self.get_layer_names(), states))
        return states


class Layer(object):
    """Super class of diverse layers."""
    def get_output_shape(self):
        return self.output_shape


class Conv2D(Layer):
    """Conv2D layer class."""
    def __init__(self, output_channels, kernel_shape, strides, padding):
        self.__dict__.update(locals())
        del self.self

    def set_input_shape(self, input_shape):
        batch_size, rows, cols, input_channels = input_shape
        kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                                   self.output_channels)
        assert len(kernel_shape) == 4
        assert all(isinstance(e, int) for e in kernel_shape), kernel_shape
        init = tf.random_normal(kernel_shape, dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init),
                                                   axis=(0, 1, 2)))
        self.kernels = tf.Variable(init)
        self.b = tf.Variable(
            np.zeros((self.output_channels,)).astype('float32'))
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = 1
        self.output_shape = tuple(output_shape)

    def fprop(self, x):
        return tf.nn.conv2d(x, self.kernels, (1,) + tuple(self.strides) + (1,),
                            self.padding) + self.b


class Linear(Layer):
    """Linear layer class."""
    def __init__(self, num_hid):
        self.num_hid = num_hid

    def set_input_shape(self, input_shape):
        batch_size, dim = input_shape
        self.input_shape = [batch_size, dim]
        self.output_shape = [batch_size, self.num_hid]
        init = tf.random_normal([dim, self.num_hid], dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0,
                                                   keepdims=True))
        self.W = tf.Variable(init)
        self.b = tf.Variable(np.zeros((self.num_hid,)).astype('float32'))

    def fprop(self, x):
        return tf.matmul(x, self.W) + self.b


class ReLU(Layer):
    """ReLU activation layer class."""
    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.relu(x)


class Sigmoid(Layer):
    """Sigmoid activation layer class."""
    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.sigmoid(x)


class Tanh(Layer):
    """Tanh activation layer class"""
    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.tanh(x)


class Softmax(Layer):
    """Softmax layer class."""
    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.softmax(x)


class Flatten(Layer):
    """Change a layer to a flattend layer."""
    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        output_width = 1
        for factor in shape[1:]:
            output_width *= factor
        self.output_width = output_width
        self.output_shape = [None, output_width]

    def fprop(self, x):
        return tf.reshape(x, [-1, self.output_width])


class MaxPool(Layer):
    def __init__(self, ksize, strides, padding):
        self.__dict__.update(locals())
        del self.self

    def set_input_shape(self, shape):
        self.input_shape = shape
        dummy_batch =tf.zeros(self.input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = 1
        self.output_shape = tuple(output_shape)

    def fprop(self, x):
        return tf.nn.max_pool(x,
                              ksize=(1,) + tuple(self.ksize) + (1,),
                              strides=(1,) + tuple(self.strides) + (1,),
                              padding=self.padding)


class Dropout(Layer):
    """Dropout layer class."""
    def __init__(self, keep_prob=0.5):
        self.keep_prob = keep_prob

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.dropout(x, keep_prob=self.keep_prob)


def mnist_cnn_model(input_shape=(None, 28, 28, 1), num_classes = 10):
    layers = [Conv2D(32, (3, 3), (1, 1), padding = "VALID"),
              ReLU(),
              Conv2D(32, (3, 3), (1, 1), padding = "VALID"),
              ReLU(),
              MaxPool((2, 2), (2, 2), padding = "VALID"),
              Conv2D(64, (3, 3), (1, 1), padding = "VALID"),
              ReLU(),
              Conv2D(64, (3, 3), (1, 1), padding = "VALID"),
              ReLU(),
              MaxPool((2, 2), (2, 2), padding = "VALID"),
              Flatten(),
              Linear(200),
              ReLU(),
              Linear(200),
              ReLU(),
              Linear(num_classes)
              ]
    model = BasicModel(layers, input_shape)
    return model