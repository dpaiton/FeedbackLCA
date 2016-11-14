## Copyright 2016 Yahoo Inc.
## Licensed under the terms of the New-BSD license.
## Please see LICENSE file in the project root for terms.
import matplotlib
matplotlib.use("Agg")

import os
import numpy as np

import models.model_constructor as mc
from data.input_data import load_MNIST

import tensorflow as tf

model_params = {
  "model_type": "LCAF",
  "model_name": "dlca_ent",
  "output_dir": os.path.expanduser("~")+"/Work/Projects/",
  "data_dir": os.path.expanduser("~")+"/Work/Datasets/MNIST/",
  "version": "0.0",
  "optimizer": "annealed_sgd",
  "auto_diff_u": True,
  "rectify_a": True,
  "norm_images": False,
  "norm_a": False,
  "norm_weights": True,
  "one_hot_labels": True,
  "batch_size": 100,
  "num_pixels": 784,
  "num_neurons": 400,
  "num_classes": 10,
  "num_val": 10000,
  "num_labeled": 50000, #[50000:1.0, 100:0.002, 20:0.0004]
  "dt": 0.001,
  "tau": 0.03,
  #TODO: The files don't stick around? Only ever have last few.
  # max to keep: https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops.html#Saver
  # set as param? Or just restructure checkpointing.
  "cp_int": 15000,
  "val_on_cp": True,
  "cp_load": True,
  "cp_load_name": "pretrain",
  "cp_load_val": 150000,
  "cp_load_ver": "0.0",
  "cp_load_var": ["phi", "bias", "w"],
  "stats_display": 10,
  "generate_plots": 50000,
  "display_plots": False,
  "save_plots": True,
  "eps": 1e-12,
  "device": "/cpu:0",
  "rand_seed": 1234567890}

model_schedule = [
  {"weights": ["phi"], # Pretrain
  "recon_mult": 1.0,
  "sparse_mult": 0.20,
  "sup_mult": 0.0,
  "fb_mult": 0.0,
  "ent_mult": 0.0,
  "num_steps": 30,
  "weight_lr": [0.15],
  "decay_steps": [30000],
  "decay_rate": [0.8],
  "staircase": [True],
  "num_batches": 150000},

  #{"weights": ["bias1", "phi", "bias2", "w"], # MLP
  #"recon_mult": 0.0,
  #"sparse_mult": 0.0,
  #"sup_mult": 1.0,
  #"fb_mult": 0.0,
  #"ent_mult": 0.0,
  #"num_steps": 20,
  #"weight_lr": [0.1]*4,
  #"decay_steps": [50000]*4,
  #"decay_rate": [0.8]*4,
  #"staircase": [True]*4,
  #"num_batches": 3000}]#,

  {"weights": ["phi", "bias", "w"], # DLCA
  "recon_mult": 1.0,
  "sparse_mult": 0.5,
  "sup_mult": 5.0,
  "fb_mult": 0.0,
  "ent_mult": 0.0,
  "num_steps": 30,
  "weight_lr": [0.0, 0.1, 0.1],
  "decay_steps": [40000]*3,
  "decay_rate": [0.8]*3,
  "staircase": [True]*3,
  "num_batches": 200000}]#,

  #{"weights": ["phi", "bias", "w"], # DLCA
  #"recon_mult": 1.0,
  #"sparse_mult": 0.5,
  #"sup_mult": 5.0,
  #"fb_mult": 0.0,
  #"ent_mult": 0.0,
  #"num_steps": 30,
  #"weight_lr": [0.0, 0.1, 0.1],
  #"decay_steps": [40000]*3,
  #"decay_rate": [0.8]*3,
  #"staircase": [True]*3,
  #"num_batches": 200000}]#,

  #{"weights": ["phi", "bias", "w"], # DLCAF
  #"recon_mult": 1.0,
  #"sparse_mult": 3.0,
  #"sup_mult": 1.0,
  #"fb_mult": 0.1,
  #"ent_mult": 0.0,
  #"num_steps": 20,
  #"weight_lr": [0.1]*3,
  #"decay_steps": [50000]*3,
  #"decay_rate": [0.8]*3,
  #"staircase": [True]*3,
  #"num_batches": 300000}]#,

  {"weights": ["phi", "bias", "w"], # DLCAF_ent
  "recon_mult": 1.0,
  "sparse_mult": 1.0,
  "sup_mult": 1.0,
  "fb_mult": 0.1,
  "ent_mult": 0.1,
  "num_steps": 20,
  "weight_lr": [0.1]*3,
  "decay_steps": [50000]*3,
  "decay_rate": [0.8]*3,
  "staircase": [True]*3,
  "num_batches": 300000}]#,

  #{"weights": ["e", "d", "g", "w", "a_bias", "c_bias"], # DRSAE
  #"recon_mult": 1.0,
  #"sparse_mult": 0.1,
  #"sup_mult": 1.0,
  #"fb_mult": 0.0,
  #"ent_mult": 0.0,
  #"num_steps": 20,
  #"weight_lr": [0.2,]*6,
  #"decay_steps": [5000,]*6,
  #"decay_rate": [0.5,]*6,
  #"staircase": [True,]*6,
  #"num_batches": 10000}]#, {}]

rand_state = np.random.RandomState(model_params["rand_seed"])

## Load data
data = load_MNIST(model_params["data_dir"],
  num_val=model_params["num_val"],
  fraction_labels=model_params["num_labeled"]/50000,
  normalize_imgs=model_params["norm_images"],
  one_hot=model_params["one_hot_labels"],
  rand_state=rand_state)

model_params["num_unlabeled"] = int(data["train"].num_examples
  - model_params["num_labeled"])

## Load model
# TODO: Write a utility to verify that model_params matches checkpoint params
#       and model_schedule matches checkpoint schedule
model_generator = getattr(mc, model_params["model_type"])
model = model_generator(model_params, model_schedule)

model.log_info("\n------------------\nTraining new model\n")
model.log_info("Training set has "+str(model_params["num_labeled"])+
  " labeled examples")
model.log_info("Model has parameter set:\n"
  +"\n".join(
  [key+"\t"+str(model_params[key])
  for key in model_params.keys()]).expandtabs(20))

model.write_saver_defs()

with tf.Session(graph=model.graph) as sess:
  ## Initialize model with empty arrays
  sess.run(model.init_op,
    feed_dict={model.s:np.zeros((model.num_pixels, model.batch_size),
    dtype=np.float32), model.y:np.zeros((model.num_classes, model.batch_size),
    dtype=np.float32)})

  model.write_graph(sess.graph_def)

  ## Load checkpoint
  if model_params["cp_load"]:
    checkpoint_file = (model.cp_load_dir+model_params["cp_load_name"]+"_v"
    +model_params["cp_load_ver"]+"_weights-"
    +str(model_params["cp_load_val"]))
    if model_params["cp_load_var"]:
      model.loader.restore(sess, checkpoint_file)
    else:
      model.weight_saver.restore(sess, checkpoint_file)
    sess.run(model.global_step.assign(model_params["cp_load_val"]))

  for sch_idx, schedule in enumerate(model_schedule):
    model.sched_idx = sch_idx
    assert model.get_sched() == schedule, ("Error: schedules do not match.")

    ## Advance data counters & skip ahead if loaded checkpoint is ahead
    if model_params["cp_load"]:
      if (data["train"].batches_completed + model.get_sched("num_batches")
        <= model_params["cp_load_val"]):
        data["train"].advance_counters(model.get_sched("num_batches"),
          model.batch_size)
        model.log_info("Skipping schedule:\n"
          +"\n".join(
          [key+"\t"+str(schedule[key])
          for key in schedule.keys()]).expandtabs(20))
        continue

    model.log_info("Beginning schedule:\n"
      +"\n".join(
      [key+"\t"+str(schedule[key])
      for key in schedule.keys()]).expandtabs(20))

    for b_step in range(model.get_sched("num_batches")):
      ## Get input batch
      mnist_batch = data["train"].next_batch(model.batch_size)
      if model_params["cp_load"]:
        if (data["train"].batches_completed
            < model_params["cp_load_val"]):
          continue
      input_images = mnist_batch[0].T
      input_labels = mnist_batch[1].T
      input_ignore_labels = mnist_batch[2].T

      ## Get feed dictionary
      feed_dict = model.get_feed_dict(input_images, input_ignore_labels)

      ## Normalize weights
      if model_params["norm_weights"]:
        sess.run(model.normalize_weights)

      ## Run inference
      if model_params["model_type"] == "LCAF":
        sess.run(model.clear_u, feed_dict)
        for u_step in range(model.get_sched("num_steps")):
          sess.run(model.step_lca, feed_dict)

      ## Update weights
      for w_idx in range(len(model.get_sched("weights"))):
        sess.run(model.apply_grads[sch_idx][w_idx], feed_dict)

      ## Generate outputs
      current_step = sess.run(model.global_step)
      if (current_step % model.stats_display == 0
        and model.stats_display > 0):
        model.print_update(input_images, input_ignore_labels, b_step+1)

      ## Validate model
      if (current_step % model.cp_int == 0
        and model.cp_int > 0):
        save_dir = model.write_checkpoint(sess)
        if model.val_on_cp:
          val_images = data["val"].images.T
          val_labels = data["val"].labels.T
          with tf.Session(graph=model.graph) as tmp_sess:
            val_feed_dict = model.get_feed_dict(val_images, val_labels)
            tmp_sess.run(model.init_op, val_feed_dict)
            model.weight_saver.restore(tmp_sess,
              save_dir+"_weights-"+str(current_step))
            if model_params["model_type"] == "LCAF":
              val_feed_dict[model.sup_mult] = 0.0
              tmp_sess.run(model.clear_u, val_feed_dict)
              for _ in range(model.get_sched("num_steps")):
                tmp_sess.run(model.step_lca, val_feed_dict)
            val_accuracy = tmp_sess.run(model.accuracy, val_feed_dict)
            model.log_info("\tvalidation accuracy: %g"%(val_accuracy))

      ## Plot weights & gradients
      if (current_step % model.gen_plots == 0
        and model.gen_plots > 0):
        model.generate_plots(input_images, input_labels)

  if model.cp_int > 0:
    save_dir = model.write_checkpoint(sess)
