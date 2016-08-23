## Copyright 2015 Yahoo Inc.
## Licensed under the terms of the New-BSD license. Please see LICENSE file in the project root for terms.
import matplotlib
matplotlib.use("Agg")

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import random

import utils.plot_functions as pf
import utils.parse_logfile
import models.model_constructor as mc
from data.input_data import load_MNIST

import tensorflow as tf

"""
Evaluate model and return activations & weights
Outputs:
  dictionary containing:
    a: activity values for specified dataset
    y_: softmax output of network for specified dataset
    phi: dictionary weights
    w: classification weights
Inputs:
  args: [dict] containing the following required keys:
    model_params: data structure specifying params, as expected by
      input/input_data.load_MNIST and models/model_constructor.Model
    model_schedule: data structure specifying schedule, as expected by
      models/model_constructor.Model
    sched_idx: [int] schedule index for the model to be tested
    checkpoint: [str] location of weights_checkpoint file to be loaded
  data: dataset to be run on, should be indexed from data/input_data.load_MNIST
    e.g. data = load_MNIST(...)["train"]

"""
def compute_inference(args, data):
  model_generator = getattr(mc, args["model_params"]["model_type"])
  model = model_generator(args["model_params"], args["model_schedule"])
  model.sched_idx = args["sched_idx"]
  out_shape = (args["num_inference_images"], args["num_inference_steps"],
    args["model_params"]["num_neurons"])
  b = np.zeros(out_shape)
  ga = np.zeros(out_shape)
  fb = np.zeros(out_shape)
  u = np.zeros(out_shape)
  a = np.zeros(out_shape)
  psnr = np.zeros((args["num_inference_images"], args["num_inference_steps"]))
  euc_loss = np.zeros((args["num_inference_images"],
    args["num_inference_steps"]))
  sparse_loss = np.zeros((args["num_inference_images"],
    args["num_inference_steps"]))
  unsup_loss = np.zeros((args["num_inference_images"],
    args["num_inference_steps"]))
  xent_loss = np.zeros((args["num_inference_images"],
    args["num_inference_steps"]))
  recon = np.zeros((args["num_inference_images"],
    args["num_inference_steps"], args["model_params"]["num_pixels"]))
  images = [None]*args["num_inference_images"]
  with tf.Session(graph=model.graph) as tmp_sess:
    feed_dict = model.get_feed_dict(np.expand_dims(data.images.T[:,0], axis=1),
        np.expand_dims(data.labels.T[:,0], axis=1))
    tmp_sess.run(model.init_op, feed_dict)
    model.weight_saver.restore(tmp_sess, args["checkpoint"])
    for img_idx in range(args["num_inference_images"]):
      image = data.next_batch(1)[0].T
      label = data.next_batch(1)[1].T
      images[img_idx] = image
      feed_dict = model.get_feed_dict(image, label)
      tmp_sess.run(model.clear_u, feed_dict)
      for step in range(args["num_inference_steps"]):
        b[img_idx, step, :] = np.squeeze(tmp_sess.run(model.lca_b, feed_dict))
        ga[img_idx, step, :] = np.squeeze(tmp_sess.run(model.lca_explain_away,
          feed_dict))
        fb[img_idx, step, :] = np.squeeze(tmp_sess.run(model.fb, feed_dict))
        u[img_idx, step, :] = np.squeeze(tmp_sess.run(model.u, feed_dict))
        a[img_idx, step, :] = np.squeeze(tmp_sess.run(model.a, feed_dict))
        psnr[img_idx, step] = np.squeeze(tmp_sess.run(model.pSNRdB, feed_dict))
        euc_loss[img_idx, step] = np.squeeze(tmp_sess.run(model.euclidean_loss,
          feed_dict))
        sparse_loss[img_idx, step] = np.squeeze(tmp_sess.run(model.sparse_loss,
          feed_dict))
        unsup_loss[img_idx, step] = np.squeeze(tmp_sess.run(
          model.unsupervised_loss, feed_dict))
        xent_loss[img_idx, step] = np.squeeze(tmp_sess.run(
          model.mean_cross_entropy_loss, feed_dict))
        recon[img_idx, step, :] = np.squeeze(tmp_sess.run(model.s_, feed_dict))
        tmp_sess.run(model.step_lca, feed_dict)
    return {"b":b, "ga":ga, "fb":fb, "u":u, "a":a, "psnr":psnr, "recon":recon,
      "sparse_loss":sparse_loss, "euc_loss":euc_loss, "unsup_loss":unsup_loss,
      "xent_loss":xent_loss, "images":np.hstack(images).T}

"""
Evaluate model and return activations & weights
Outputs:
  dictionary containing:
    a: activity values for specified dataset
    y_: softmax output of network for specified dataset
    phi: dictionary weights
    w: classification weights
Inputs:
  args: [dict] containing the following required keys:
    model_params: data structure specifying params, as expected by
      input/input_data.load_MNIST and models/model_constructor.Model
    model_schedule: data structure specifying schedule, as expected by
      models/model_constructor.Model
    sched_idx: [int] schedule index for the model to be tested
    checkpoint: [str] location of weights_checkpoint file to be loaded
  data: dataset to be run on, should be indexed from data/input_data.load_MNIST
    e.g. data = load_MNIST(...)["train"]

"""
def evaluate_model(args, data):
  images = data.images.T
  labels = data.labels.T
  model_generator = getattr(mc, args["model_params"]["model_type"])
  model = model_generator(args["model_params"], args["model_schedule"])
  model.sched_idx = args["sched_idx"]
  feed_dict = model.get_feed_dict(images, labels)
  num_imgs = images.shape[1]
  with tf.Session(graph=model.graph) as tmp_sess:
    tmp_sess.run(model.init_op, feed_dict)
    model.weight_saver.restore(tmp_sess, args["checkpoint"])
    if model.model_type == "LCAF":
      tmp_sess.run(model.clear_u, feed_dict)
      for _ in range(model.get_sched("num_steps")):
        tmp_sess.run(model.step_lca, feed_dict)
    a = tmp_sess.run(model.a, feed_dict)
    phi = tmp_sess.run(tf.transpose(model.phi))
    w = tmp_sess.run(model.w)
    s_ = tmp_sess.run(model.s_, feed_dict)
    y_ = tmp_sess.run(model.y_, feed_dict)
    pSNRdB = tmp_sess.run(model.pSNRdB, feed_dict)
  return {"a":a, "phi":phi, "w":w, "s_":s_, "y_":y_, "pSNRdB":pSNRdB}

"""
Compute model performance on an MNIST dataset
Outputs:
  tuple containing:
    accuracy: [np.ndarray] containing accuracy values
    cross_entropy: [np.ndarray] containing cross-entropy loss values
    sparsity: [np.ndarray] containing sparseness values
    reconstruction: [np.ndarray] containing reconstruction quality values
Inputs:
  args: [dict] containing the following required keys:
    model_params: data structure specifying params, as expected by
      input/input_data.load_MNIST and models/model_constructor.Model
    model_schedule: data structure specifying schedule, as expected by
      models/model_constructor.Model
    sched_idx: [int] schedule index for the model to be tested
    plot_sem: [bool] if set, plot error bars for the standard error of the mean
    checkpoint: [str] location of weights_checkpoint file to be loaded
"""
def compute_performance(args, data):
  images = data.images.T
  labels = data.labels.T
  model_generator = getattr(mc, args["model_params"]["model_type"])
  model = model_generator(args["model_params"], args["model_schedule"])
  model.sched_idx = args["sched_idx"]
  num_imgs = images.shape[1]
  with tf.Session(graph=model.graph) as tmp_sess:
    if args["plot_sem"]:
      # Compute accuracy one image at a time to get SEM
      tmp_sess.run(model.init_op,
        feed_dict={
        model.s:np.expand_dims(images[:, 0], axis=1),
        model.y:np.expand_dims(labels[:, 0], axis=1)})
      num_runs = num_imgs
    else:
      tmp_sess.run(model.init_op, feed_dict={model.s:images, model.y:labels})
      num_runs = 1
    accuracy = np.zeros(num_runs)
    cross_entropy = np.zeros(num_runs)
    sparsity = np.zeros(num_runs)
    reconstruction = np.zeros(num_runs)
    model.weight_saver.restore(tmp_sess, args["checkpoint"])
    for run_idx in range(num_runs):
      if args["plot_sem"]:
        image_set = np.expand_dims(images[:, run_idx], axis=1)
        label_set = np.expand_dims(labels[:, run_idx], axis=1)
      else:
        image_set = images
        label_set = labels
      feed_dict = model.get_feed_dict(image_set, label_set)
      if model.model_type == "LCAF":
        feed_dict[model.fb_mult] = 0.0
        tmp_sess.run(model.clear_u, feed_dict)
        for _ in range(model.get_sched("num_steps")):
          tmp_sess.run(model.step_lca, feed_dict)
      if hasattr(model, "accuracy"):
        accuracy[run_idx] = tmp_sess.run(model.accuracy, feed_dict)
      else:
        accuracy[run_idx] = 0
      if hasattr(model, "cross_entropy_loss"):
        cross_entropy[run_idx] = tmp_sess.run(model.mean_cross_entropy_loss,
          feed_dict)
      else:
        cross_entropy[run_idx] = 0
      if hasattr(model, "sparse_loss"):
        sparsity[run_idx] = tmp_sess.run(model.sparse_loss, feed_dict)
      else:
        sparsity[run_idx] = 0
      if hasattr(model, "pSNRdB"):
        reconstruction[run_idx] = tmp_sess.run(model.pSNRdB, feed_dict)
      else:
        reconstruction[run_idx] = 0
  return (accuracy, cross_entropy, sparsity, reconstruction)

"""
Entry point for analysis
Inputs:
  args: [dict] containing the following required keys:
    run_test: [bool] run analysis on test data
    run_val: [bool] run analysis on validation data
    params: [list] of model params, where each element in the list is the
      data structure that would be required by input/input_data.load_MNIST
      and models/model_constructor.Model
    schedule: [list] of model schedules, where each element in the list is
      the data structure that would be required by
      models/model_constructor.Model
    loss: [list] containing outputs from utils/parse_logfiles.read_loss()
    checkpoint_dir: [str] location of weights_checkpoint file to be loaded
    analysis_dir: [str] location of analysis output directory
"""
def main(args):
  num_conditions = len(args["params"])
  assert num_conditions == len(args["schedule"])
  assert num_conditions == len(args["loss"])
  if args["run_test"]:
    test_performance = [None]*num_conditions
  if args["run_val"]:
    val_performance = [None]*num_conditions

  mnist_data = load_MNIST(args["params"][0]["data_dir"],
    num_val=10000,
    fraction_labels=1.0,
    normalize_imgs=True,
    one_hot=args["params"][0]["one_hot_labels"],
    rand_seed=args["params"][0]["rand_seed"])

  ## Analysis per condition (frac labeled examples)
  loss = [None] * num_conditions
  recon = [None] * num_conditions
  for condition_idx in range(num_conditions):
    print("Computing performance evaluations for run %g out of %g"%(
      condition_idx+1, num_conditions))

    params = args["params"][condition_idx]
    sched = args["schedule"][condition_idx]
    loss[condition_idx] = args["loss"][condition_idx]

    base_filename = (args["analysis_dir"]+params["model_name"]+"_v"
      +params["version"])

    ## Generate plots for loss values over time
    loss_filename = base_filename+"_loss"+args["file_ext"]
    acc_filename = base_filename+"_accuracy"+args["file_ext"]
    err_filename = base_filename+"_reconstruction"+args["file_ext"]
    pf.save_losses(loss[condition_idx], loss_filename)
    pf.save_accuracy(loss[condition_idx], acc_filename)
    if "recon_quality" in loss[condition_idx].keys():
      pf.save_recon(loss[condition_idx], err_filename)

    ## Compute test & val performance for trained model
    chk_args = dict()
    sched_idx = len(sched)-1
    batch_idx = 0
    for schedule in sched:
      batch_idx += schedule["num_batches"]
    args["checkpoint"] = (args["checkpoint_dir"]+params["model_name"]
      +"_v"+params["version"]+"_full-"
      +str(batch_idx))
    print("\tLoading file "+args["checkpoint"])
    args["model_params"] = params
    args["model_schedule"] = sched
    args["sched_idx"] = sched_idx
    if args["run_test"]:
      test_performance[condition_idx] = compute_performance(args,
        mnist_data["test"])
    if args["run_val"]:
      val_performance[condition_idx] = compute_performance(args,
        mnist_data["val"])

    ## Plot inference for a single display period
    if args["inference"] and params["model_type"] == "LCAF":
      inference_data = compute_inference(args, mnist_data["train"])
      pf.save_inference_stats(inference_data, base_filename, args["file_ext"],
        num_skip=10)
      pf.save_inference_traces(inference_data, base_filename, args["file_ext"])

    if args["eval_train"]:
      ## Evaluate model on dataset for future analysis
      train = evaluate_model(args, mnist_data["train"])

      ## Generate top & bottom phi elements associated with each class in w
      sorted_w = np.argsort(train["w"]).astype(np.int32)
      top_phi = np.zeros((params["num_classes"], args["num_phi"],
        params["num_pixels"]), dtype=np.float32)
      bot_phi = np.zeros((params["num_classes"], args["num_phi"],
        params["num_pixels"]), dtype=np.float32)
      #TODO: Write out function to construct a single properly formatted plot
      for c in np.arange(params["num_classes"]):
        top_phi[c, ...] = train["phi"][sorted_w[c,::-1][0:args["num_phi"]], :]
        bot_phi[c, ...] = train["phi"][sorted_w[c,:][0:args["num_phi"]], :]
        data = np.vstack((top_phi[c, ...], bot_phi[c, ...]))
        data = data.reshape((2*args["num_phi"],
          int(np.sqrt(params["num_pixels"])),
          int(np.sqrt(params["num_pixels"]))))
        title = "Top & Bottom phi Elements for Digit "+str(c)
        out_filename = base_filename+"_top_phi_"+str(c)+args["file_ext"]
        normalize = True
        pf.save_data_tiled(data, normalize, title, out_filename)

      ## Frequency of activation per element across dataset
      perc_used = (100.0 * np.sort(np.sum(train["a"] != 0, axis=1))[::-1]
        / train["a"].shape[1])
      act_freq_out_filename = (base_filename+"_train_activation_frequency"
        +args["file_ext"])
      xlabel = "Basis Element"
      ylabel = "Percent of Representations"
      title = ("Percent of Image Representations that Requried a Given"
        +" Basis Element")
      xticklabels = ["" for _ in range(len(perc_used))]
      pf.save_bar(perc_used, xticklabels=xticklabels,
          out_filename=act_freq_out_filename, xlabel=xlabel, ylabel=ylabel,
          title=title)

      ## Average activity across dataset
      act_out_filename = (base_filename+"_train_avg_activation"
        +args["file_ext"])
      xlabel = "Element"
      ylabel = "Average Activation"
      title = ("Average Element Activation Amplitude Across the Training Set")
      avg_act = np.sort(np.mean(train["a"], axis=1))[::-1]
      xticklabels = ["" for _ in range(len(avg_act))]
      pf.save_bar(avg_act, xticklabels=xticklabels,
        out_filename=act_out_filename, xlabel=xlabel, ylabel=ylabel,
        title=title)

      ## Visualization of dictionary sorted by associated dataset activity
      sorted_act_idx = np.argsort(np.sum(train["a"] != 0, axis=1))[::-1]
      weights_sorted = train["phi"][sorted_act_idx, :].reshape(
        train["phi"].shape[0], int(np.sqrt(train["phi"].shape[1])),
        int(np.sqrt(train["phi"].shape[1])))
      weight_out_filename = (base_filename+"_sorted_weights"+args["file_ext"])
      fig_title = "Phi sorted by activation frequency in training set"
      normalize = True
      pf.save_data_tiled(weights_sorted, normalize, fig_title,
        weight_out_filename)

      ## Compute entropy of class predictions across dataset
      ent = -np.sum(train["y_"] * np.log(train["y_"]+1e-16), axis=0)
      print(("\tThe mean entropy of y_ on the train set was %g with STD %g.")%(
        np.mean(ent), np.std(ent)))

      ## Mean sparsity across the training set
      mean_sparsity = (100 * np.mean([np.count_nonzero(train["a"][:,idx])
        for idx in range(train["a"].shape[1])]) / float(train["a"].shape[0]))
      print(("\tThe mean sparsity on the train set was %0.2f%%")%mean_sparsity)

      ## Mean reconstruction quality across the training set
      mean_recon_quality = np.mean(train["pSNRdB"])
      print(("\tThe mean reconstruction peak SNR on the train set was %0.2f dB")%(
        mean_recon_quality))

  ## Compute average error rate on Val & Test sets
  if args["run_test"] or args["run_val"]:
    base_filename = (args["analysis_dir"]+params["model_name"]+"_v"
      +params["base_version"])
    num_samples = [params["num_labeled"] for params in args["params"]]
    error_rates = []
    error_str = []
    if args["run_test"]:
      num_test = 10000
      if args["plot_sem"]:
        test_out_filename = (base_filename+"_performance_sem"
          +args["file_ext"])
      else:
        test_out_filename = (base_filename+"_performance"+args["file_ext"])
      test_fig = pf.save_performance(test_performance, num_samples,
        test_out_filename, "test", args["plot_sem"])
      error_rates.append([num_test * (1 - test_performance[idx][0][0])
        for idx in np.arange(len(test_performance))])
      error_str.append("test")

    if args["run_val"]:
      num_val = 10000
      if args["plot_sem"]:
        val_out_filename = base_filename+"_performance_sem"+args["file_ext"]
      else:
        val_out_filename = base_filename+"_performance"+args["file_ext"]
      val_fig = pf.save_performance(val_performance, num_samples,
        val_out_filename, "val", args["plot_sem"])
      error_rates.append([num_val * (1-val_performance[idx][0][0])
        for idx in np.arange(len(val_performance))])
      error_str.append("val")

    for err_idx, error_rate in enumerate(error_rates):
      out_str = "\n".join([("%s\t\t||  %d\t\t||  %0.2f\t\t||  %0.2f")%(
        num_samples[idx], int(error_rate[idx]), np.mean(loss[idx]["perc_act"]),
        np.mean(loss[idx]["recon_quality"]))
        for idx in np.arange(len(num_samples))])
      print("-".join(["" for _ in range(75)]))
      print(("Num Labeled Ex\t||  %s Error\t||  Mean Perc Active\t||  "
        +"Recon Quality\n%s")%(error_str[err_idx], out_str))
      print("-".join(["" for _ in range(75)]))

  ## Final outputs
  print("Analysis complete for version "+str(params["base_version"])+"\n")

if __name__ == "__main__":
  args = dict()

  #versions = [str(val) for val in range(0,8)]
  versions = ["10"]

  args["model_name"] = "pretrain"
  #args["model_name"] = "nofb" # nofb_sup
  #args["model_name"] = "nofb_sup_nopre"
  #args["model_name"] = "nofb_semisup"
  #args["model_name"] = "nofb_semisup_nopre"
  #args["model_name"] = "fb_semisup"
  #args["model_name"] = "fb_semisup_nopre"
  #args["model_name"] = "feedforward_semisup"
  #args["model_name"] = "mlp"

  args["eval_train"] = True # Evaluate model stats on training set
  args["plot_sem"] = False # Plot SEM bars when able
  args["run_test"] = False # Evaluate model accuracy on test set
  args["run_val"] = False # Evaluate model accuracy on validation set
  args["inference"] = True # Evaluate LCA inference

  args["num_phi"] = 8 # How many phi to view for a given w connection
  args["num_inference_images"] = 5 # How many images in average 
  args["num_inference_steps"] = 30 # How many time steps in inference

  args["file_ext"] = ".pdf" # Output file format
  args["device"] = "/cpu:0" # Device for analysis runs

  for version in versions:
    args["log_file"] = (os.path.expanduser("~")
      +"/Work/Projects/"+args["model_name"]+"/logfiles/"+args["model_name"]
      +"_v"+version+".log")
    log_text = utils.parse_logfile.load_file(args["log_file"])
    run_text = re.split("------------------", log_text)[1:]
    args["params"] = []
    args["schedule"] = []
    args["loss"] = []
    for idx, text in enumerate(run_text):
      args["params"].append(utils.parse_logfile.read_params(text))
      args["params"][idx]["device"] = args["device"]
      args["params"][idx]["data_dir"] = (
        os.path.expanduser("~")+"/Work/Datasets/MNIST/")
      args["schedule"].append(utils.parse_logfile.read_schedule(text))
      args["loss"].append(utils.parse_logfile.read_loss(text))
      args["params"][idx]["output_dir"] = (os.path.expanduser("~")
        +"/Work/Projects/")

    args["analysis_dir"] = (args["params"][0]["output_dir"]
      +args["params"][0]["model_name"]+"/analysis/")
    args["checkpoint_dir"] = (args["params"][0]["output_dir"]
      +args["params"][0]["model_name"]+"/checkpoints/")

    tf.set_random_seed(args["params"][0]["rand_seed"])
    np.random.seed(args["params"][0]["rand_seed"])
    random.seed(args["params"][0]["rand_seed"])

    main(args)
