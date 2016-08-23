## Copyright 2016 Yahoo Inc.
## Licensed under the terms of the New-BSD license.
## Please see LICENSE file in the project root for terms.
import os
import logging
import numpy as np
import utils.plot_functions as pf
import tensorflow as tf

"""
Modifiable Parameters:
  model_type     [str] Type of model. Can be "Model", "LCAF" or "Feedforward"
  model_name     [str] Name for model
  output_dir     [str] Base directory where output will be directed
  base_version   [str] Unmodified base version for output
  optimizer      [str] Which optimization algorithm to use
                       Can be "annealed_sgd" or "adam"
  auto_diff_u    [bool] LCAF - If set, use auto-differentiation for u update
  norm_a         [bool] LCAF - If set, l2 normalize layer 1 activity
  norm_weights   [bool] If set, l2 normalize weights after updates
  one_hot_labels [bool] If set, use one-hot labels
  batch_size     [int] Number of images in a training batch
  num_pixels     [int] Number of pixels
  num_neurons    [int] Number of layer 1 elements (# basis vectors)
  num_classes    [int] Number of layer 2 elements (# categories)
  num_val        [int] Number of validation images
  dt             [float] LCAF - Discrete global time constant
  tau            [float] LCAF - LCA time constant
  cp_int         [int] How often to checkpoint
  val_on_cp      [bool] If set, compute validation performance on checkpoint
  cp_load        [bool] if set, load from checkpoint
  cp_load_name   [str] checkpoint model name to load
  cp_load_val    [int] checkpoint time step to load
  cp_load_ver    [str] checkpoint version to load
  stats_display  [int] How often to send updates to stdout
  generate_plots [int] How often to generate plots
  display_plots  [bool] If set, display plots
  save_plots     [bool] If set, save plots to file
  eps            [float] Small value to avoid division by zero
  device         [str] Which device to run on
  rand_seed      [int] Random seed

Schedule options:
  weights     [list of str] Which weights to update
  recon_mult  [float] Unsupervised loss tradeoff
  sparse_mult [float] Sparsity tradeoff
  ent_mult    [float] Entropy loss tradeoff
  sup_mult    [float] Supervised loss tradeoff (cross-entropy)
  fb_mult     [float] Supervised feedback strength (cross-entropy)
  num_steps   [int] Number of iterations of LCA update equation
  weight_lr   [list of float] Learning rates for weight updates
  decay_steps [list of int] How often to decay for SGD annealing
  decay_rate  [list of float] Rate to decay for SGD annealing
  staircase   [list of bool] Annealing step (T) or exponential (F)
  num_batches [int] Number of batches to run for this schedule
"""
class Model(object):
  def __init__(self, params, schedule):
    self.graph_built = False
    self.optimizers_added = False
    self.savers_constructed = False
    self.load_schedule(schedule)
    self.sched_idx = 0
    self.load_params(params)
    if self.rand_seed is not None:
      tf.set_random_seed(self.rand_seed)
      np.random.seed(self.rand_seed)
    self.init_logging()
    self.make_dirs()
    self.build_graph()
    self.add_optimizers_to_graph()
    self.add_initializer_to_graph()
    self.construct_savers()
    self.recon_prev_fig = None
    self.w_prev_fig = None
    self.phi_prev_fig = None

  """
  Load schedule into object
  Inputs:
   schedule: [list of dict] learning schedule
  """
  def load_schedule(self, schedule):
    for sched in schedule:
      assert len(sched["weights"]) == len(sched["weight_lr"])
      assert len(sched["weights"]) == len(sched["decay_steps"])
      assert len(sched["weights"]) == len(sched["decay_rate"])
      assert len(sched["weights"]) == len(sched["staircase"])
    self.sched = schedule

  """
  Load parameters into object
  Inputs:
   params: [dict] model parameters
  """
  def load_params(self, params):
    # Meta-parameters
    self.model_name = str(params["model_name"])
    self.model_type = str(params["model_type"])
    if "num_labeled" in params.keys():
      self.num_labeled = str(params["num_labeled"])
    if "num_unlabeled" in params.keys():
      self.num_unlabeled = str(params["num_unlabeled"])
    self.base_version = str(params["base_version"])
    if "version" in params.keys():
      self.version = str(params["version"])
    else:
      self.version = str(params["base_version"])
    self.optimizer = str(params["optimizer"])
    self.rectify_a = bool(params["rectify_a"])
    self.norm_a = bool(params["norm_a"])
    self.norm_weights = bool(params["norm_weights"])
    self.one_hot_labels = bool(params["one_hot_labels"])
    assert self.one_hot_labels, (
      "One-hot labels are currently the only type supported.")
    # Hyper-parameters
    self.batch_size = int(params["batch_size"])
    self.num_pixels = int(params["num_pixels"])
    self.num_neurons = int(params["num_neurons"])
    self.num_classes = int(params["num_classes"])
    self.num_val = int(params["num_val"])
    self.phi_shape = [self.num_pixels, self.num_neurons]
    self.w_shape = [self.num_classes, self.num_neurons]
    # Output generation
    self.stats_display = int(params["stats_display"])
    self.val_on_cp = bool(params["val_on_cp"])
    self.gen_plots = int(params["generate_plots"])
    self.disp_plots = bool(params["display_plots"])
    self.save_plots = bool(params["save_plots"])
    # Checkpoints
    self.cp_int = int(params["cp_int"])
    self.cp_load = bool(params["cp_load"])
    self.cp_load_name = str(params["cp_load_name"])
    self.cp_load_val = int(params["cp_load_val"])
    self.cp_load_ver = str(params["cp_load_ver"])
    # Directories
    self.out_dir = str(params["output_dir"]) + self.model_name
    self.cp_save_dir = self.out_dir + "/checkpoints/"
    self.cp_load_dir = (str(params["output_dir"]) + self.cp_load_name
      + "/checkpoints/")
    self.log_dir = self.out_dir + "/logfiles/"
    self.save_dir = self.out_dir + "/savefiles/"
    self.disp_dir = self.out_dir + "/vis/"
    self.analysis_dir = self.out_dir + "/analysis/"
    # Other
    self.eps = float(params["eps"])
    self.device = str(params["device"])
    self.rand_seed = int(params["rand_seed"])

  """Logging to std:err to track run duration"""
  def init_logging(self):
    logging_level = logging.INFO
    log_format = ("%(asctime)s.%(msecs)03d"
      +" -- %(message)s")
    logging.basicConfig(format=log_format, datefmt="%H:%M:%S",
      level=logging.INFO)

  """Make output directories"""
  def make_dirs(self):
    if not os.path.exists(self.out_dir):
      os.makedirs(self.out_dir)
    if not os.path.exists(self.log_dir):
      os.makedirs(self.log_dir)
    if not os.path.exists(self.cp_save_dir):
      os.makedirs(self.cp_save_dir)
    if not os.path.exists(self.save_dir):
      os.makedirs(self.save_dir)
    if not os.path.exists(self.disp_dir):
      os.makedirs(self.disp_dir)
    if not os.path.exists(self.analysis_dir):
      os.makedirs(self.analysis_dir)

  """
  Build the TensorFlow graph object.
  The default graph is an MLP.
  """
  def build_graph(self):
    self.graph = tf.Graph()
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("placeholders") as scope:
          self.s = tf.placeholder(tf.float32,
            shape=[self.num_pixels, None], name="input_data")
          self.y = tf.placeholder(tf.float32,
            shape=[self.num_classes, None], name="input_label")
          self.sparse_mult = tf.placeholder(
            tf.float32, shape=(), name="sparse_mult")
          self.recon_mult = tf.placeholder(
            tf.float32, shape=(), name="recon_mult")
          self.sup_mult = tf.placeholder(
            tf.float32, shape=(), name="sup_mult")
          self.ent_mult = tf.placeholder(
            tf.float32, shape=(), name="ent_mult")

        with tf.name_scope("constants") as scope:
          self.label_mult = tf.reduce_sum(self.y, reduction_indices=[0])

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False,
            name="global_step")

        with tf.variable_scope("weights") as scope:
          self.phi = tf.get_variable(name="phi", dtype=tf.float32,
            initializer=tf.truncated_normal(self.phi_shape, mean=0.0,
            stddev=1.0, dtype=tf.float32, name="phi_init"), trainable=True)
          self.w = tf.get_variable(name="w", dtype=tf.float32,
            initializer=tf.truncated_normal(self.w_shape, mean=0.0,
            stddev=1.0, dtype=tf.float32, name="w_init"), trainable=True)
          self.b = tf.get_variable(name="bias", dtype=tf.float32,
            initializer=tf.zeros([self.num_classes, 1], dtype=tf.float32,
            name="bias_init"), trainable=True)

        with tf.name_scope("normalize_weights") as scope:
          self.norm_phi = self.phi.assign(tf.nn.l2_normalize(self.phi,
            dim=0, epsilon=self.eps, name="row_l2_norm"))
          self.normalize_weights = tf.group(self.norm_phi,
            name="do_normalization")

        with tf.name_scope("hidden_variables") as scope:
          if self.rectify_a:
            self.a = tf.nn.relu(tf.matmul(tf.transpose(self.phi), self.s),
              name="activity")
          else:
            self.a = tf.matmul(tf.transpose(self.phi), self.s, name="activity")

          if self.norm_a:
            self.c = tf.add(tf.matmul(self.w, tf.nn.l2_normalize(self.a,
              dim=0, epsilon=self.eps, name="row_l2_norm"),
              name="classify"), self.b, name="c")
          else:
            self.c = tf.add(tf.matmul(self.w, self.a, name="classify"), self.b,
              name="c")

        with tf.name_scope("output") as scope:
          with tf.name_scope("image_estimate"):
            self.s_ = tf.matmul(self.phi, self.a, name="reconstruction")
          with tf.name_scope("label_estimate"):
            self.y_ = tf.transpose(tf.nn.softmax(tf.transpose(self.c)))

        with tf.name_scope("loss") as scope:
          with tf.name_scope("unsupervised"):
            self.euclidean_loss = self.recon_mult * tf.reduce_mean(0.5 *
              tf.reduce_sum(tf.pow(tf.sub(self.s, self.s_), 2.0),
              reduction_indices=[0]))
            self.sparse_loss = self.sparse_mult * tf.reduce_mean(
              tf.reduce_sum(tf.abs(self.a), reduction_indices=[0]))
            self.entropy_loss = self.ent_mult * tf.reduce_mean(
              -tf.reduce_sum(tf.mul(tf.clip_by_value(self.y_, self.eps, 1.0),
              tf.log(tf.clip_by_value(self.y_, self.eps, 1.0))),
              reduction_indices=[0]))
            self.unsupervised_loss = (self.euclidean_loss + self.sparse_loss)

          with tf.name_scope("supervised"):
            with tf.name_scope("cross_entropy_loss"):
              label_count = tf.reduce_sum(self.label_mult)
              self.cross_entropy_loss = self.sup_mult * (
                tf.reduce_sum(self.label_mult * -tf.reduce_sum(tf.mul(self.y,
                tf.log(tf.clip_by_value(self.y_, self.eps, 1.0))),
                reduction_indices=[0])) / label_count)
            self.supervised_loss = self.cross_entropy_loss
          self.total_loss = self.unsupervised_loss + self.supervised_loss

        with tf.name_scope("performance_metrics") as scope:
          with tf.name_scope("reconstruction_quality"):
            MSE = tf.reduce_mean(tf.pow(tf.sub(self.s, self.s_), 2.0),
              name="mean_squared_error")
            self.pSNRdB = tf.mul(10.0, tf.log(tf.div(tf.pow(255.0, 2.0), MSE)),
              name="recon_quality")
          with tf.name_scope("prediction_bools"):
            self.correct_prediction = tf.equal(tf.argmax(self.y_, dimension=0),
              tf.argmax(self.y, dimension=0), name="individual_accuracy")
          with tf.name_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
              tf.float32), name="avg_accuracy")

    self.graph_built = True

  """
  Add optimizers to graph
  Creates member variables grads_and_vars and apply_grads for each weight
    - both member variables are indexed by [schedule_idx][weight_idx]
    - grads_and_vars holds the gradients for the weight updates
    - apply_grads is the operator to be called to perform weight updates
  """
  def add_optimizers_to_graph(self):
    with self.graph.as_default():
      with tf.name_scope("optimizers") as scope:
        self.grads_and_vars = list()
        self.apply_grads = list()
        for sch_idx, sch in enumerate(self.sched):
          sch_grads_and_vars = list()
          sch_apply_grads = list()
          for w_idx, weight in enumerate(sch["weights"]):
            learning_rates = tf.train.exponential_decay(
              learning_rate=sch["weight_lr"][w_idx],
              global_step=self.global_step,
              decay_steps=sch["decay_steps"][w_idx],
              decay_rate=sch["decay_rate"][w_idx],
              staircase=sch["staircase"][w_idx],
              name="annealing_schedule_"+weight)
            if self.optimizer == "annealed_sgd":
              optimizer = tf.train.GradientDescentOptimizer(learning_rates,
                name="grad_optimizer_"+weight)
            elif self.optimizer == "adam":
              optimizer = tf.train.AdamOptimizer(learning_rates,
                beta1=0.9, beta2=0.99, epsilon=1e-07,
                name="adam_optimizer_"+weight)
            with tf.variable_scope("weights", reuse=True) as scope:
              weight_var = [tf.get_variable(weight)]
            sch_grads_and_vars.append(
              optimizer.compute_gradients(self.total_loss, var_list=weight_var))
            if w_idx == 0: # Only want to update global step once
              sch_apply_grads.append(
                optimizer.apply_gradients(sch_grads_and_vars[w_idx],
                global_step=self.global_step))
            else:
              sch_apply_grads.append(
                optimizer.apply_gradients(sch_grads_and_vars[w_idx],
                global_step=None))
          self.grads_and_vars.append(sch_grads_and_vars)
          self.apply_grads.append(sch_apply_grads)
    self.optimizers_added = True

  """Add initializer to the graph
  This must be done after optimizers have been added
  """
  def add_initializer_to_graph(self):
    assert self.graph_built
    assert self.optimizers_added
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("initialization") as scope:
          self.init_op = tf.initialize_all_variables()

  """Add savers to graph"""
  def construct_savers(self):
    assert self.graph_built, (
      "The graph must be built before adding savers.")
    assert self.optimizers_added, (
      "Optimizers must be added to the graph before constructing savers.")
    with self.graph.as_default():
      with tf.variable_scope("weights", reuse=True) as scope:
        weights = [weight for weight in tf.all_variables()
          if weight.name.startswith(scope.name)]
      self.weight_saver = tf.train.Saver(weights)
      self.full_saver = tf.train.Saver()
    self.savers_constructed = True

  """Write saver definitions for full model and weights-only"""
  def write_saver_defs(self):
    assert self.savers_constructed
    full_saver_def = self.full_saver.as_saver_def()
    full_file = self.save_dir+self.model_name+"_v"+self.version+"-full.def"
    with open(full_file, "wb") as f:
        f.write(full_saver_def.SerializeToString())
    logging.info("Full saver def saved in file %s"%full_file)
    weight_saver_def = self.weight_saver.as_saver_def()
    weight_file = self.save_dir+self.model_name+"_v"+self.version+"-weights.def"
    with open(weight_file, "wb") as f:
        f.write(weight_saver_def.SerializeToString())
    logging.info("Weight saver def saved in file %s"%weight_file)

  """Write graph structure to protobuf file"""
  def write_graph(self, graph_def):
    write_name = self.model_name+"_v"+self.version+".pb"
    self.writer = tf.train.SummaryWriter(self.save_dir, graph=self.graph)
    tf.train.write_graph(graph_def,
      logdir=self.save_dir, name=write_name, as_text=False)
    logging.info("Graph def saved in file %s"%self.save_dir+write_name)

  """Write checkpoints for full model and weights-only"""
  def write_checkpoint(self, session):
    base_save_path = self.cp_save_dir+self.model_name+"_v"+self.version
    full_save_path = self.full_saver.save(session,
      save_path=base_save_path+"_full", global_step=self.global_step)
    logging.info("Full model saved in file %s"%full_save_path)
    weight_save_path = self.weight_saver.save(session,
      save_path=base_save_path+"_weights",
      global_step=self.global_step)
    logging.info("Weights model saved in file %s"%weight_save_path)
    return base_save_path

  """
  Returns the current schedule being executed
  Inputs:
    key: [str] key in dictionary
  """
  def get_sched(self, key=None):
    if key:
      assert key in self.sched[self.sched_idx].keys(), (
        key+" must be in the schedule.")
      return self.sched[self.sched_idx][key]
    return self.sched[self.sched_idx]

  """
  Modifies the internal schedule for the current schedule index
  Inputs:
    key: [str] key in dictionary
    val: value be set in schedlue,
      if there is already a val for key, new val must be of same type
  """
  def set_sched(self, key, val):
    if key in self.sched[self.sched_idx].keys():
      assert type(val) == type(self.sched[self.sched_idx][key]), (
        "val must have type "+str(type(self.sched[self.sched_idx][key])))
    self.sched[self.sched_idx][key] = val

  """
  Load checkpoint weights into session.
  Inputs:
    session: tf.Session() that you want to load into
    model_dir: String specifying the path to the checkpoint
  """
  def load_model(self, session, model_dir):
    self.full_saver.restore(session, model_dir)

  """Use logging to print input string to stderr"""
  def log_info(self, string):
    logging.info(str(string))

  """
  Return dictionary containing all placeholders
  Inputs:
    input_data: data to be placed in self.s
    input_label: label to be placed in self.y
  """
  def get_feed_dict(self, input_data, input_labels):
    placeholders = [op.name
      for op
      in self.graph.get_operations()
      if "placeholders" in op.name][2:]
    feed_dict = {self.s:input_data, self.y:input_labels}
    for placeholder in placeholders:
      feed_dict[self.graph.get_tensor_by_name(placeholder+":0")] = (
        self.get_sched(placeholder.split('/')[1]))
    return feed_dict

  """
  Log train progress information
  Inputs:
    input_data: load_MNIST data object containing the current image batch
    input_label: load_MNIST data object containing the current label batch
    batch_step: current batch number within the schedule
  """
  def print_update(self, input_data, input_label, batch_step):
    current_step = self.global_step.eval()
    feed_dict = self.get_feed_dict(input_data, input_label)
    a_vals = self.a.eval(feed_dict)
    logging.info("Global batch index is %g"%(current_step))
    logging.info("Finished step %g out of %g for schedule %g"%(batch_step,
      self.get_sched("num_batches"), self.sched_idx))
    logging.info(
      "\teuclidean loss:\t\t%g"%(self.euclidean_loss.eval(feed_dict)))
    logging.info("\tsparse loss:\t\t%g"%(self.sparse_loss.eval(feed_dict)))
    logging.info(
      "\tunsupervised loss:\t%g"%(self.unsupervised_loss.eval(feed_dict)))
    logging.info("\tsupervised loss:\t%g"%(
      self.supervised_loss.eval(feed_dict)))
    logging.info("\tmax val of a:\t\t%g"%(a_vals.max()))
    logging.info("\tpercent active:\t\t%0.2f%%"%(
      100.0 * np.count_nonzero(a_vals)
      / float(self.num_neurons * self.batch_size)))
    logging.info("\trecon pSNR dB:\t\t%g"%(self.pSNRdB.eval(feed_dict)))
    logging.info("\ttrain accuracy:\t\t%g"%(self.accuracy.eval(feed_dict)))

  """
  Plot weights, reconstruction, and gradients
  Inputs: input_data and input_label used for the session
  """
  def generate_plots(self, input_image, input_label):
    feed_dict = self.get_feed_dict(input_image, input_label)
    current_step = str(self.global_step.eval())
    if self.disp_plots:
      self.recon_prev_fig = pf.display_data_tiled(
        tf.transpose(self.s_).eval(feed_dict).reshape(
        self.batch_size, int(np.sqrt(self.num_pixels)),
        int(np.sqrt(self.num_pixels))), title=("Reconstructions at step "
        +current_step), prev_fig=self.recon_prev_fig)
      self.w_prev_fig = pf.display_data_tiled(
        self.w.eval().reshape(self.num_classes,
        int(np.sqrt(self.num_neurons)), int(np.sqrt(self.num_neurons))),
        title="Classification matrix at step number "+current_step,
        prev_fig=self.w_prev_fig)
      self.phi_prev_fig = pf.display_data_tiled(
        tf.transpose(self.phi).eval().reshape(self.num_neurons,
        int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
        title="Dictionary at step "+current_step,
        prev_fig=self.phi_prev_fig)
    if self.save_plots:
      pf.save_data_tiled(
        tf.transpose(self.s_).eval(feed_dict).reshape(
        self.batch_size, int(np.sqrt(self.num_pixels)),
        int(np.sqrt(self.num_pixels))), normalize=False,
        title=("Reconstructions at step "+current_step),
        save_filename=(self.disp_dir+"recon_v"+self.version+"-"
        +current_step.zfill(5)+".pdf"))
      pf.save_data_tiled(
        self.w.eval().reshape(self.num_classes,
        int(np.sqrt(self.num_neurons)), int(np.sqrt(self.num_neurons))),
        normalize=True, title="Classification matrix at step number "
        +current_step, save_filename=(self.disp_dir+"w_v"+self.version+"-"
        +current_step.zfill(5)+".pdf"))
      pf.save_data_tiled(
        tf.transpose(self.phi).eval().reshape(self.num_neurons,
        int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
        normalize=True, title="Dictionary at step "+current_step,
        save_filename=(self.disp_dir+"phi_v"+self.version+"-"
        +current_step.zfill(5)+".pdf"))
      for weight_grad_var in self.grads_and_vars[self.sched_idx]:
        grad = weight_grad_var[0][0].eval(feed_dict)
        shape = grad.shape
        name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]
        if name == "phi":
          pf.save_data_tiled(grad.T.reshape(self.num_neurons,
            int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
            normalize=True, title="Gradient for phi at step "+current_step,
            save_filename=(self.disp_dir+"dphi_v"+self.version+"_"
            +current_step.zfill(5)+".pdf"))
        elif name == "w":
          pf.save_data_tiled(grad.reshape(self.num_classes,
            int(np.sqrt(self.num_neurons)), int(np.sqrt(self.num_neurons))),
            normalize=True, title="Gradient for w at step "+current_step,
            save_filename=(self.disp_dir+"dw_v"+self.version+"_"
            +current_step.zfill(5)+".pdf"))

"""
DRSAE model implemented after:
  JT Rolfe, Y Lecun (2013) -
  "Discriminative Recurrent Sparse Auto-Encoders"
"""
class DRSAE(Model):
  def __init__(self, params, schedule):
    Model.__init__(self, params, schedule)
    self.e_prev_fig = None
    self.d_prev_fig = None
    self.g_prev_fig = None

  def load_params(self, params):
    Model.load_params(self, params)

  """Build the TensorFlow graph object"""
  def build_graph(self):
    self.graph = tf.Graph()
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("placeholders") as scope:
          self.s = tf.placeholder(
            tf.float32, shape=[self.num_pixels, None], name="input_data")
          self.y = tf.placeholder(
            tf.float32, shape=[self.num_classes, None], name="input_label")
          self.sparse_mult = tf.placeholder(
            tf.float32, shape=(), name="sparse_mult")
          self.recon_mult = tf.placeholder(
            tf.float32, shape=(), name="recon_mult")
          self.sup_mult = tf.placeholder(
            tf.float32, shape=(), name="sup_mult")
          self.ent_mult = tf.placeholder(
            tf.float32, shape=(), name="ent_mult")

        with tf.name_scope("constants") as scope:
          self.label_mult = tf.reduce_sum(self.y, reduction_indices=[0])
          self.a_zeros = tf.zeros(
            shape=tf.pack([self.num_neurons, tf.shape(self.s)[1]]),
            dtype=tf.float32, name="a_zeros")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.variable_scope("weights") as scope:
          self.e = tf.get_variable(name="e", dtype=tf.float32,
            initializer=tf.truncated_normal([self.num_neurons, self.num_pixels],
            mean=0.0, stddev=1.0, dtype=tf.float32, name="e_init"),
            trainable=True)
          self.d = tf.get_variable(name="d", dtype=tf.float32,
            initializer=tf.truncated_normal([self.num_pixels, self.num_neurons],
            mean=0.0, stddev=1.0, dtype=tf.float32, name="d_init"),
            trainable=True)
          self.g = tf.get_variable(name="g", dtype=tf.float32,
            initializer=tf.truncated_normal(
            [self.num_neurons, self.num_neurons], mean=0.0,
            stddev=1.0, dtype=tf.float32, name="g_init"), trainable=True)
          self.w = tf.get_variable(name="w", dtype=tf.float32,
            initializer=tf.truncated_normal(self.w_shape, mean=0.0,
            stddev=1.0, dtype=tf.float32, name="w_init"), trainable=True)
          self.a_bias = tf.get_variable(name="a_bias", dtype=tf.float32,
            initializer=tf.zeros([self.num_neurons, 1], dtype=tf.float32,
            name="a_bias_init"), trainable=True)
          self.c_bias = tf.get_variable(name="c_bias", dtype=tf.float32,
            initializer=tf.zeros([self.num_classes, 1], dtype=tf.float32,
            name="c_bias_init"), trainable=True)

        with tf.name_scope("normalize_weights") as scope:
          self.norm_e = self.e.assign(tf.nn.l2_normalize(self.e,
            dim=1, epsilon=self.eps, name="col_l2_norm"))
          self.norm_d = self.d.assign(tf.nn.l2_normalize(self.d,
            dim=0, epsilon=self.eps, name="row_l2_norm"))
          self.normalize_weights = tf.group(self.norm_e, self.norm_d,
            name="do_normalization")

        with tf.name_scope("dynamic_variable") as scope:
          self.a = tf.Variable(self.a_zeros, trainable=False,
            validate_shape=False, name="a")

        with tf.name_scope("update_a") as scope:
          self.da = (tf.matmul(self.e, self.s, name="encoding_transform")
            + tf.matmul(self.g, self.a, name="explaining_away") - self.a_bias)
          self.step_a = tf.group(self.a.assign(self.da), name="do_update_a")
          self.clear_a = tf.group(self.a.assign(self.a_zeros),
            name="do_clear_a")

        with tf.name_scope("classifier") as scope:
          if self.norm_a:
            self.c = tf.add(tf.matmul(self.w, tf.nn.l2_normalize(self.a,
              dim=0, epsilon=self.eps, name="col_l2_norm"),
              name="classify"), self.c_bias)
          else:
            self.c = tf.add(tf.matmul(self.w, self.a, name="classify"),
              self.c_bias)

        with tf.name_scope("output") as scope:
          with tf.name_scope("image_estimate"):
            self.s_ = tf.matmul(self.d, self.a, name="reconstruction")
          with tf.name_scope("label_estimate"):
            self.y_ = tf.transpose(tf.nn.softmax(tf.transpose(self.c)))

        with tf.name_scope("loss") as scope:
          with tf.name_scope("unsupervised"):
            self.euclidean_loss = self.recon_mult * tf.reduce_mean(0.5 *
              tf.reduce_sum(tf.pow(tf.sub(self.s, self.s_), 2.0),
              reduction_indices=[0]))
            self.sparse_loss = self.sparse_mult * tf.reduce_mean(
              tf.reduce_sum(tf.abs(self.a), reduction_indices=[0]))
            self.entropy_loss = self.ent_mult * tf.reduce_mean(
              -tf.reduce_sum(tf.mul(tf.clip_by_value(self.y_, self.eps, 1.0),
              tf.log(tf.clip_by_value(self.y_, self.eps, 1.0))),
              reduction_indices=[0]))
            self.unsupervised_loss = (self.euclidean_loss + self.sparse_loss)

          with tf.name_scope("supervised"):
            with tf.name_scope("cross_entropy_loss"):
              label_count = tf.reduce_sum(self.label_mult)
              self.cross_entropy_loss = self.sup_mult * (
                tf.reduce_sum(self.label_mult * -tf.reduce_sum(tf.mul(self.y,
                tf.log(tf.clip_by_value(self.y_, self.eps, 1.0))),
                reduction_indices=[0])) / label_count)
            self.supervised_loss = self.cross_entropy_loss
          self.total_loss = self.unsupervised_loss + self.supervised_loss

        with tf.name_scope("performance_metrics") as scope:
          with tf.name_scope("reconstruction_quality"):
            MSE = tf.reduce_mean(tf.pow(tf.sub(self.s, self.s_), 2.0),
              name="mean_squared_error")
            self.pSNRdB = tf.mul(10.0, tf.log(tf.div(tf.pow(255.0, 2.0), MSE)),
              name="recon_quality")
          with tf.name_scope("prediction_bools"):
            self.correct_prediction = tf.equal(tf.argmax(self.y_, dimension=0),
              tf.argmax(self.y, dimension=0), name="individual_accuracy")
          with tf.name_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
              tf.float32), name="avg_accuracy")

    self.graph_built = True

  """
  Plot weights, reconstruction, and gradients
  Inputs: input_data and input_label used for the session
  """
  def generate_plots(self, input_image, input_label):
    feed_dict = self.get_feed_dict(input_image, input_label)
    current_step = str(self.global_step.eval())
    if self.disp_plots:
      self.recon_prev_fig = pf.display_data_tiled(
        tf.transpose(self.s_).eval(feed_dict).reshape(
        self.batch_size, int(np.sqrt(self.num_pixels)),
        int(np.sqrt(self.num_pixels))), title=("Reconstructions at step "
        +current_step), prev_fig=self.recon_prev_fig)
      self.w_prev_fig = pf.display_data_tiled(
        self.w.eval().reshape(self.num_classes,
        int(np.sqrt(self.num_neurons)), int(np.sqrt(self.num_neurons))),
        title="Classification matrix at step number "+current_step,
        prev_fig=self.w_prev_fig)
      self.e_prev_fig = pf.display_data_tiled(
        self.e.eval().reshape(self.num_neurons,
        int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
        title="Encoding dictionary at step "+current_step,
        prev_fig=self.e_prev_fig)
      self.d_prev_fig = pf.display_data_tiled(
        tf.transpose(self.d).eval().reshape(self.num_neurons,
        int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
        title="Decoding dictionary at step "+current_step,
        prev_fig=self.d_prev_fig)
      self.g_prev_fig = pf.display_data_tiled(
        self.g.eval().reshape(self.num_neurons*self.num_neurons,
        int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
        title="Explaining Away dictionary at step "+current_step,
        prev_fig=self.g_prev_fig)
    if self.save_plots:
      pf.save_data_tiled(
        tf.transpose(self.s_).eval(feed_dict).reshape(
        self.batch_size, int(np.sqrt(self.num_pixels)),
        int(np.sqrt(self.num_pixels))), normalize=False,
        title=("Reconstructions at step "+current_step),
        save_filename=(self.disp_dir+"recon_v"+self.version+"-"
        +current_step.zfill(5)+".pdf"))
      pf.save_data_tiled(
        self.w.eval().reshape(self.num_classes,
        int(np.sqrt(self.num_neurons)), int(np.sqrt(self.num_neurons))),
        normalize=True, title="Classification matrix at step number "
        +current_step, save_filename=(self.disp_dir+"w_v"+self.version+"-"
        +current_step.zfill(5)+".pdf"))
      pf.save_data_tiled(
        self.e.eval().reshape(self.num_neurons,
        int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
        normalize=True, title="Encoding dictionary at step "+current_step,
        save_filename=(self.disp_dir+"e_v"+self.version+"-"
        +current_step.zfill(5)+".pdf")
      pf.save_data_tiled(
        tf.transpose(self.d).eval().reshape(self.num_neurons,
        int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
        normalize=True, title="Decoding dictionary at step "+current_step,
        save_filename=(self.disp_dir+"d_v"+self.version+"-"
        +current_step.zfill(5)+".pdf")
      pf.save_data_tiled(
        self.g.eval().reshape(self.num_neurons*self.num_neurons,
        int(np.sqrt(self.num_pixels)), int(np.sqrt(self.num_pixels))),
        normalize=True, title="Explaining Away dictionary at step "
        +current_step, save_filename=(self.disp_dir+"g_v"+self.version+"-"
        +current_step.zfill(5)+".pdf")
      for weight_grad_var in self.grads_and_vars[self.sched_idx]:
        grad = weight_grad_var[0][0].eval(feed_dict)
        shape = grad.shape
        name = weight_grad_var[0][1].name.split('/')[1].split(':')[0]
        if name == "w":
          pf.save_data_tiled(grad.reshape(self.num_classes,
            int(np.sqrt(self.num_neurons)), int(np.sqrt(self.num_neurons))),
            title="Gradient for w at step "+current_step,
            save_filename=(self.disp_dir+"dw_v"+self.version+"_"
            +current_step.zfill(5)+".pdf"))

"""
LCAF model is an extension of the LCA model described in:
  CJ Rozell, DH Johnson, RG Baraniuk, BA Olshausen (2008) -
  "Sparse Coding via Thresholding and Local Competition in Neural Circuits"

Extended model includes a classification layer for supervised training
and modification to the inference process to include feedback from the
classification layer.
"""
class LCAF(Model):
  def __init__(self, params, schedule):
    Model.__init__(self, params, schedule)

  def load_params(self, params):
    Model.load_params(self, params)
    self.dt = float(params["dt"])
    self.tau = float(params["tau"])
    self.eta = self.dt / self.tau
    self.auto_diff_u = bool(params["auto_diff_u"])

  """Build the TensorFlow graph object"""
  def build_graph(self):
    self.graph = tf.Graph()
    with tf.device(self.device):
      with self.graph.as_default():
        with tf.name_scope("placeholders") as scope:
          self.s = tf.placeholder(
            tf.float32, shape=[self.num_pixels, None], name="input_data")
          self.y = tf.placeholder(
            tf.float32, shape=[self.num_classes, None], name="input_label")
          self.sparse_mult = tf.placeholder(
            tf.float32, shape=(), name="sparse_mult")
          self.recon_mult = tf.placeholder(
            tf.float32, shape=(), name="recon_mult")
          self.sup_mult = tf.placeholder(
            tf.float32, shape=(), name="sup_mult")
          self.ent_mult = tf.placeholder(
            tf.float32, shape=(), name="ent_mult")
          self.fb_mult = tf.placeholder(
            tf.float32, shape=(), name="fb_mult")

        with tf.name_scope("constants") as scope:
          self.u_zeros = tf.zeros(
            shape=tf.pack([self.num_neurons, tf.shape(self.s)[1]]),
            dtype=tf.float32, name="u_zeros")
          self.label_mult = tf.reduce_sum(self.y, reduction_indices=[0],
            name="label_multiplier")

        with tf.name_scope("step_counter") as scope:
          self.global_step = tf.Variable(0, trainable=False, name="global_step")

        with tf.variable_scope("weights") as scope:
          self.phi = tf.get_variable(name="phi", dtype=tf.float32,
            initializer=tf.truncated_normal(self.phi_shape, mean=0.0,
            stddev=1.0, dtype=tf.float32, name="phi_init"), trainable=True)
          self.w = tf.get_variable(name="w", dtype=tf.float32,
            initializer=tf.truncated_normal(self.w_shape, mean=0.0,
            stddev=1.0, dtype=tf.float32, name="w_init"), trainable=True)
          self.b = tf.get_variable(name="bias", dtype=tf.float32,
            initializer=tf.zeros([self.num_classes, 1], dtype=tf.float32,
            name="bias_init"), trainable=True)

        with tf.name_scope("normalize_weights") as scope:
          self.norm_phi = self.phi.assign(tf.nn.l2_normalize(self.phi,
            dim=0, epsilon=self.eps, name="row_l2_norm"))
          self.normalize_weights = tf.group(self.norm_phi,
            name="do_normalization")

        with tf.name_scope("inference") as scope:
          self.u = tf.Variable(self.u_zeros, trainable=False,
            validate_shape=False, name="u")
          if self.rectify_a:
            self.a = tf.select(tf.greater(self.u, self.sparse_mult),
              tf.sub(self.u, self.sparse_mult), self.u_zeros, name="activity")
          else:
            self.a = tf.select(tf.greater(self.u, self.sparse_mult),
              tf.sub(self.u, self.sparse_mult), tf.select(tf.less(self.u,
              -self.sparse_mult), tf.add(self.u, self.sparse_mult),
              self.u_zeros), name="activity")

        with tf.name_scope("classifier") as scope:
          if self.norm_a:
            self.c = tf.add(tf.matmul(self.w, tf.nn.l2_normalize(self.a,
              dim=0, epsilon=self.eps, name="col_l2_norm"),
              name="classify"), self.b)
          else:
            self.c = tf.add(tf.matmul(self.w, self.a, name="classify"), self.b)

        with tf.name_scope("output") as scope:
          with tf.name_scope("image_estimate"):
            self.s_ = tf.matmul(self.phi, self.a, name="reconstruction")
          with tf.name_scope("label_estimate"):
            self.y_ = tf.clip_by_value(tf.transpose(tf.nn.softmax(
              tf.transpose(self.c))), self.eps, 1.0)

        with tf.name_scope("loss") as scope:
          with tf.name_scope("unsupervised"):
            self.euclidean_loss = self.recon_mult * tf.reduce_mean(0.5 *
              tf.reduce_sum(tf.pow(tf.sub(self.s, self.s_), 2.0),
              reduction_indices=[0]))
            self.sparse_loss = self.sparse_mult * tf.reduce_mean(
              tf.reduce_sum(tf.abs(self.a), reduction_indices=[0]))
            self.entropy_loss = -tf.reduce_sum(tf.mul(self.y_,
              tf.log(self.y_)), reduction_indices=[0])
            self.mean_entropy_loss = (self.ent_mult * tf.reduce_mean(
              self.entropy_loss, name="mean_entropy_loss"))
            self.unsupervised_loss = (self.euclidean_loss + self.sparse_loss)

          with tf.name_scope("supervised"):
            with tf.name_scope("cross_entropy_loss"):
              self.cross_entropy_loss = (self.label_mult
                * -tf.reduce_sum(tf.mul(self.y, tf.log(self.y_)),
                reduction_indices=[0]))
              label_count = tf.reduce_sum(self.label_mult)
              self.mean_cross_entropy_loss = (self.sup_mult
                * tf.reduce_sum(self.cross_entropy_loss) /
                (label_count + self.eps))
            self.supervised_loss = self.mean_cross_entropy_loss
          self.total_loss = self.unsupervised_loss + self.supervised_loss

        with tf.name_scope("update_u") as scope:
          self.lca_b = tf.matmul(tf.transpose(self.phi), self.s,
            name="driving_input")
          self.lca_g = (tf.matmul(tf.transpose(self.phi), self.phi,
            name="gram_matrix") -
            tf.constant(np.identity(int(self.phi_shape[1]), dtype=np.float32),
            name="identity_matrix"))
          self.lca_explain_away = tf.matmul(self.lca_g, self.a,
            name="explaining_away")

          if self.auto_diff_u:
            self.fb = self.fb_mult * (
              self.sup_mult * tf.gradients(self.cross_entropy_loss, self.a)[0]
              + (1-self.label_mult) * self.ent_mult
              * tf.gradients(self.entropy_loss, self.a)[0])
          else:
            self.fb = (self.sup_mult * self.fb_mult * self.label_mult
            * tf.matmul(tf.transpose(self.w), tf.sub(self.y_, self.y)))

          self.du = (self.lca_b - self.lca_explain_away - self.u - self.fb)

          self.step_lca = tf.group(self.u.assign_add(self.eta * self.du),
            name="do_update_u")
          self.clear_u = tf.group(self.u.assign(self.u_zeros),
            name="do_clear_u")

        with tf.name_scope("performance_metrics") as scope:
          with tf.name_scope("reconstruction_quality"):
            MSE = tf.reduce_mean(tf.pow(tf.sub(self.s, self.s_), 2.0),
              name="mean_squared_error")
            self.pSNRdB = tf.mul(10.0, tf.log(tf.div(tf.pow(255.0, 2.0), MSE)),
              name="recon_quality")
          with tf.name_scope("prediction_bools"):
            self.correct_prediction = tf.equal(tf.argmax(self.y_, dimension=0),
              tf.argmax(self.y, dimension=0), name="individual_accuracy")
          with tf.name_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
              tf.float32), name="avg_accuracy")

    self.graph_built = True
