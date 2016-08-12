## Copyright 2015 Yahoo Inc.
## Licensed under the terms of the New-BSD license. Please see LICENSE file in the project root for terms.
import os
from tensorflow.python.client import graph_util
import tensorflow as tf

class checkpoint_session:
  def __init__(self, params):
    if not tf.gfile.Exists(params["graph"]):
      print("Input graph file '" + params["graph"] + "' does not exist!")
      return -1

    if not tf.gfile.Exists(params["saver"]):
      print("Input saver file '" + params["saver"] + "' does not exist!")
      return -1

    if not tf.gfile.Glob(params["checkpoint"]):
      print("Input checkpoint '" + params["checkpoint"] + "' doesn't exist!")
      return -1

    self.params = params
    self.session = None
    self.graph_def = None
    self.node_names = []
    self.loaded = False

  def load(self):
    ## Read binary graph file into graph_def structure
    self.graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(self.params["graph"], "rb") as f:
      self.graph_def.ParseFromString(f.read())

    ## Strip nodes of device specification, collect list of node names
    for node in self.graph_def.node:
      node.device = ""
      self.node_names.append(node.name)

    ## Load the session graph
    _ = tf.import_graph_def(self.graph_def, name="")

    ## Initialize the session, restore variables from checkpoint
    self.session = tf.Session()
    with tf.gfile.FastGFile(self.params["saver"], "rb") as f:
      saver_def = tf.train.SaverDef()
      saver_def.ParseFromString(f.read())
      saver = tf.train.Saver(saver_def=saver_def)
      saver.restore(self.session, self.params["checkpoint"])

    self.loaded = True

  def write_constant_graph_def(self):
    output_graph_def = graph_util.convert_variables_to_constants(self.session,
      self.graph_def, self.node_names)
    with tf.gfile.GFile(params["graph"]+".frozen", "wb") as f:
      f.write(output_graph_def.SerializeToString())

  def close_sess(self):
    self.session.close()
