import os
import numpy
from ns_cifar10.utils import data_cifar10

class DataSet(object):
  def __init__(self, images, labels):
      assert images.shape[0] == labels.shape[0], (
        "images.shape: %s labels.shape: %s" % (images.shape, labels.shape))
      self._num_examples = images.shape[0]
      self._images = images
      self._labels = labels
      self._epochs_completed = 0
      self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def read_cifar10_data_sets(data_dir, one_hot=True):
  class DataSets(object):
    pass
  data_sets = DataSets()

  train_images, train_labels, test_images, test_labels = data_cifar10(datadir="./data/cifar10",
                                                                      train_start=0, train_end=50000,
                                                                      test_start=0, test_end=10000)
  data_sets.train = DataSet(train_images, train_labels)
  data_sets.test = DataSet(test_images, test_labels)
  return data_sets


def read_adv_data_sets(advdata_dir, is_targeted=False):
  class DataSets(object):
    pass
  data_sets = DataSets()
  if is_targeted:
    ADV_EXAMPLES = 'adversarial-examples-targeted.npy'
  else:
    ADV_EXAMPLES = 'adversarial-examples.npy'
  TRUE_LABELS = 'true-labels.npy'
  adv_examples = numpy.load(os.path.join(advdata_dir, ADV_EXAMPLES))
  true_labels = numpy.load(os.path.join(advdata_dir, TRUE_LABELS))
  data_sets.data = DataSet(adv_examples, true_labels)
  return data_sets