## Copyright 2015 Yahoo Inc.
## Licensed under the terms of the New-BSD license. Please see LICENSE file in the project root for terms.
import numpy as np
import random
import gzip

class MNIST:
  def __init__(self,
    img_dir,
    lbl_dir,
    num_val=10000,
    fraction_labels=1.0,
    one_hot=True,
    rand_seed=None):

    if num_val < 1:
      num_val = 0
    if fraction_labels > 1.0:
      fraction_labels = 1.0
    if fraction_labels < 0.0:
      fraction_labels = 0.0

    self.num_classes = 10 # 10 MNIST classes

    ## Extract images
    self.images = self.extract_images(img_dir)
    self.labels = self.extract_labels(lbl_dir)
    if one_hot:
      self.labels_1h = self.dense_to_one_hot(self.labels)
    assert self.images.shape[0] == self.labels.shape[0], (
      "Error: %g images and %g labels"%(self.images.shape[0],
      self.labels.shape[0]))

    ## Grab a random sample of images for the validation set
    tot_imgs = self.images.shape[0]
    if tot_imgs < num_val:
      num_val = tot_imgs
    if num_val > 0:
      self.val_indices = np.random.choice(np.arange(tot_imgs, dtype=np.int32),
        size=num_val, replace=False)
      self.img_indices = np.setdiff1d(np.arange(tot_imgs, dtype=np.int32),
        self.val_indices).astype(np.int32)
    else:
      self.val_indices = None
      self.img_indices = np.arange(tot_imgs, dtype=np.int32)

    self.num_imgs = len(self.img_indices)
    self.num_keep = int(self.num_imgs * float(fraction_labels))

    ## Construct list of images to be ignored
    if self.num_keep < self.num_imgs:
      self.ignore_indices = []
      for lbl in range(0, self.num_classes):
        lbl_loc = [idx
          for idx
          in np.arange(len(self.img_indices), dtype=np.int32)
          if self.labels[self.img_indices[idx]] == lbl]
        np.append(self.ignore_indices, np.random.choice(lbl_loc, 
          size=int(len(lbl_loc) - (self.num_keep/float(self.num_classes))),
          replace=False))
    else:
      self.ignore_indices = None

  def dense_to_one_hot(self, labels_dense):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels, dtype=np.int32) * self.num_classes
    labels_one_hot = np.zeros((num_labels, self.num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

  def read_4B(self, bytestream):
    dt = np.dtype(np.uint32).newbyteorder("B") #big-endian byte order- MSB first
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

  def read_img_header(self, bytestream):
    _ = self.read_4B(bytestream)
    num_images = self.read_4B(bytestream)
    img_rows = self.read_4B(bytestream)
    img_cols = self.read_4B(bytestream)
    return (num_images, img_rows, img_cols)

  def read_lbl_header(self, bytestream):
    _ = self.read_4B(bytestream)
    return self.read_4B(bytestream)

  def extract_images(self, filename):
    with open(filename, "rb") as f:
      with gzip.GzipFile(fileobj=f) as bytestream:
        num_images, img_rows, img_cols = self.read_img_header(bytestream)
        buf = bytestream.read(num_images*img_rows*img_cols)
        images = np.frombuffer(buf, dtype=np.uint8)
        return images.reshape(num_images, img_rows, img_cols).astype(np.float32)

  def extract_labels(self, filename):
    with open(filename, "rb") as f:
      with gzip.GzipFile(fileobj=f) as bytestream:
        num_labels = self.read_lbl_header(bytestream)
        buf = bytestream.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels.astype(np.int32)

class dataset:
  def __init__(self, imgs, lbls, ignore_lbls, normalize=False):
    num_ex = imgs.shape[0]
    num_rows = imgs.shape[1]
    num_cols = imgs.shape[2]
    if normalize:
      self.images = self.normalize_image(imgs.reshape(num_ex,
        num_rows*num_cols))
    else:
      self.images = imgs.reshape(num_ex, num_rows*num_cols)
      self.images /= 255.0
    self.labels = lbls
    self.ignore_labels = ignore_lbls
    self.num_examples = num_ex
    self.epochs_completed = 0
    self.batches_completed = 0
    self.curr_epoch_idx = 0
    self.epoch_order = np.random.permutation(self.num_examples)

  """
  Advance epoch counter & generate new index order
  Inputs:
    num_to_advance : [int] number of epochs to advance
  """
  def new_epoch(self, num_to_advance=1):
    self.epochs_completed += int(num_to_advance)
    for _ in range(int(num_to_advance)):
      self.epoch_order = np.random.permutation(self.num_examples)

  """
  Return a batch of images
  Outputs:
    3d tuple containing images, labels and ignore labels
  Inputs:
    batch_size : [int] representing the number of images in the batch
  Function assumes that batch_size is a scalar increment of num_examples.
  """
  def next_batch(self, batch_size):
    assert batch_size < self.num_examples, (
      "Error: Batch size cannot be greater than"
      +"the number of examples in an epoch")
    if self.num_examples % batch_size != 0:
      print("data/input_data.py: WARNING: batch_size should divide evenly into"
       +" num_examples. Some images may not be included in the dataset.")
    if self.curr_epoch_idx + batch_size > self.num_examples:
      start = 0
      self.new_epoch(1)
      self.curr_epoch_idx = 0
    else:
      start = self.curr_epoch_idx
    self.batches_completed += 1
    self.curr_epoch_idx += batch_size
    return (self.images[self.epoch_order[start:self.curr_epoch_idx], ...],
      self.labels[self.epoch_order[start:self.curr_epoch_idx], ...],
      self.ignore_labels[self.epoch_order[start:self.curr_epoch_idx], ...])

  """
  Increment member variables to reflect a step forward of num_advance images
  Inputs:
    num_batches: How many batches to step forward
    batch_size: How many examples constitute a batch
  """
  def advance_counters(self, num_batches, batch_size):
    if num_batches * batch_size > self.num_examples:
      self.new_epoch(int((num_batches * batch_size) / float(self.num_examples)))
    self.batches_completed += num_batches
    self.curr_epoch_idx = (num_batches * batch_size) % self.num_examples

  """
  Normalize input image to have mean 0 and std 1
  The operation is done per image, not across the batch.
  Outputs:
    norm: normalized image
  Inputs:
    img: numpy ndarray of dim [num_batch, num_data]
  """
  def normalize_image(self, img):
    norm = np.vstack([(img[idx,:]-np.mean(img[idx,:]))/np.std(img[idx,:])
      for idx
      in range(img.shape[0])])
    return norm

def load_MNIST(
  data_dir,
  num_val=10000,
  fraction_labels=1.0,
  normalize_imgs=False,
  one_hot=True,
  rand_seed=None):

  ## Training set
  train_img_filename = data_dir+"/train-images-idx3-ubyte.gz"
  train_lbl_filename = data_dir+"/train-labels-idx1-ubyte.gz"
  train_val = MNIST(
    train_img_filename,
    train_lbl_filename,
    num_val=num_val,
    fraction_labels=fraction_labels,
    one_hot=one_hot,
    rand_seed=rand_seed)
  train_imgs = train_val.images[train_val.img_indices, ...]
  if one_hot:
    train_lbls = train_val.labels_1h[train_val.img_indices, ...]
    train_ignore_lbls = train_lbls.copy()
    if train_val.ignore_indices is not None:
      train_ignore_lbls[train_val.ignore_indices, ...] = 0
  else:
    train_lbls = train_val.labels[train_val.img_indices]
    train_ignore_lbls = train_lbls.copy()
    if train_val.ignore_indices is not None:
      train_ignore_lbls[train_val.ignore_indices] = -1
  train = dataset(train_imgs, train_lbls, train_ignore_lbls,
    normalize=normalize_imgs)

  ## Validation set
  if num_val > 0:
    val_imgs = train_val.images[train_val.val_indices]
    if one_hot:
      val_lbls = train_val.labels_1h[train_val.val_indices]
    else:
      val_lbls = train_val.labels[train_val.val_indices]
    val_ignore_lbls = val_lbls.copy()
    val = dataset(val_imgs, val_lbls, val_ignore_lbls, normalize=normalize_imgs)
  else:
    val = None

  ## Test set
  test_img_filename = data_dir+"/t10k-images-idx3-ubyte.gz"
  test_lbl_filename = data_dir+"/t10k-labels-idx1-ubyte.gz"
  test = MNIST(
    test_img_filename,
    test_lbl_filename,
    num_val=0,
    fraction_labels=1.0,
    one_hot=one_hot,
    rand_seed=rand_seed)
  test_imgs = test.images
  if one_hot:
    test_lbls = test.labels_1h
  else:
    test_lbls = test.labels
  test_ignore_lbls = test_lbls.copy()
  test = dataset(test_imgs, test_lbls, test_ignore_lbls,
    normalize=normalize_imgs)

  return {"train":train, "val":val, "test":test}
