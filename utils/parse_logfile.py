## Copyright 2015 Yahoo Inc.
## Licensed under the terms of the New-BSD license. Please see LICENSE file in the project root for terms.
import re
import numpy as np

"""
Load loss file into memory
Output: string containing log file text
Input: string containing the filename of the log file
"""
def load_file(log_file):
  with open(log_file, "r") as f:
    log_text = f.read()
  return log_text

"""
Generate dictionary of model parameters
  Input: string containing log file text
  Output: Dictionary of model parameters
"""
def read_params(log_text):
  model_type = (
    re.findall("model_type\s+(\w+)", log_text))
  model_name = (
    re.findall("model_name\s+(\w+)", log_text))
  output_dir = (
    re.findall("output_dir\s+([\w\/]+)", log_text))
  data_dir = (
    re.findall("data_dir\s+([\w\/]+)", log_text))
  base_version = (
    re.findall("base_version\s+(\d+.?\d*)", log_text))
  version = (
    re.findall("[^_]version\s+([\d.?]+)", log_text))
  optimizer = (
    re.findall("optimizer\s+(\w+)", log_text))
  auto_diff_u = (
    [val=="True" for val in re.findall("auto_diff_u\s+(\w+)", log_text)])
  norm_a = (
    [val=="True" for val in re.findall("norm_a\s+(\w+)", log_text)])
  rectify_a = (
    [val=="True" for val in re.findall("rectify_a\s+(\w+)", log_text)])
  norm_weights = (
    [val=="True" for val in re.findall("norm_weights\s+(\w+)", log_text)])
  one_hot_labels = (
    [val=="True" for val in re.findall("one_hot_labels\s+(\w+)", log_text)])
  batch_size = (
    [int(val) for val in re.findall("batch_size\s+(\d+)", log_text)])
  num_pixels = (
    [int(val) for val in re.findall("num_pixels\s+(\d+)", log_text)])
  num_neurons = (
    [int(val) for val in re.findall("num_neurons\s+(\d+)", log_text)])
  num_classes = (
    [int(val) for val in re.findall("num_classes\s+(\d+)", log_text)])
  num_val = (
    [int(val) for val in re.findall("num_val\s+(\d+)", log_text)])
  num_labeled = (re.findall("num_labeled\s+(\d+)", log_text))
  num_unlabeled = (re.findall("num_unlabeled\s+(\d+)", log_text))
  dt = (
    [float(val) for val in re.findall("dt\s+(\d+\.?\d+)", log_text)])
  tau = (
    [float(val) for val in re.findall("tau\s+(\d+\.?\d+)", log_text)])
  cp_int = (
    [int(val) for val in re.findall("cp_int\s+(\d+)", log_text)])
  val_on_cp = (
    [val=="True" for val in re.findall("val_on_cp\s+(\w+)", log_text)])
  cp_load = (
    [val=="True" for val in re.findall("cp_load\s+(\w+)", log_text)])
  cp_load_name = (
    [str(val) for val in re.findall("cp_load_name\s+(\w+)", log_text)])
  cp_load_val = (
    [int(val) for val in re.findall("cp_load_val\s+(\d+)", log_text)])
  cp_load_ver = (
    [str(val) for val in re.findall("cp_load_ver\s+(\w+)", log_text)])
  stats_display = (
    [int(val) for val in re.findall("stats_display\s+(\d+)", log_text)])
  generate_plots = (
    [int(val) for val in re.findall("generate_plots\s+(-?\d+)", log_text)])
  display_plots = (
    [val=="True" for val in re.findall("display_plots\s+(\w+)", log_text)])
  save_plots = (
    [val=="True" for val in re.findall("save_plots\s+(\w+)", log_text)])
  eps = (
    [float(val)
    for val in re.findall("eps\s+(\d+\.?\d*[[Ee]\-?\d+]?)", log_text)])
  device = (
    [str(val) for val in re.findall("device\s+(\/\w+\:\d)", log_text)])
  rand_seed = (
    [int(val) for val in re.findall("rand_seed\s+(\d+)", log_text)])
  output = {
    "model_name": model_name if len(model_name)>1 else model_name[0],
    "model_type": model_type if len(model_type)>1 else model_type[0],
    "output_dir": output_dir if len(output_dir)>1 else output_dir[0],
    "data_dir": data_dir if len(data_dir)>1 else data_dir[0],
    "base_version": base_version if len(base_version)>1 else base_version[0],
    "version": version if len(version)>1 else version[0],
    "optimizer": optimizer if len(optimizer)>1 else optimizer[0],
    "auto_diff_u": auto_diff_u if len(auto_diff_u)>1 else auto_diff_u[0],
    "norm_a": norm_a if len(norm_a)>1 else norm_a[0],
    "rectify_a": rectify_a if len(rectify_a)>1 else rectify_a[0],
    "norm_weights": norm_weights if len(norm_weights)>1 else norm_weights[0],
    "one_hot_labels": one_hot_labels
      if len(one_hot_labels)>1 else one_hot_labels[0],
    "batch_size": batch_size if len(batch_size)>1 else batch_size[0],
    "num_pixels": num_pixels if len(num_pixels)>1 else num_pixels[0],
    "num_neurons": num_neurons if len(num_neurons)>1 else num_neurons[0],
    "num_classes": num_classes if len(num_classes)>1 else num_classes[0],
    "num_val": num_val if len(num_val)>1 else num_val[0],
    "num_labeled": num_labeled if len(num_labeled)>1 else num_labeled[0],
    "num_unlabeled": num_unlabeled
      if len(num_unlabeled)>1 else num_unlabeled[0],
    "dt": dt if len(dt)>1 else dt[0],
    "tau": tau if len(tau)>1 else tau[0],
    "cp_int": cp_int if len(cp_int)>1 else cp_int[0],
    "val_on_cp": val_on_cp if len(val_on_cp)>1 else val_on_cp[0],
    "cp_load": cp_load if len(cp_load)>1 else cp_load[0],
    "cp_load_name": cp_load_name if len(cp_load_name)>1 else cp_load_name[0],
    "cp_load_val": cp_load_val if len(cp_load_val)>1 else cp_load_val[0],
    "cp_load_ver": cp_load_ver if len(cp_load_ver)>1 else cp_load_ver[0],
    "stats_display": stats_display
      if len(stats_display)>1 else stats_display[0],
    "generate_plots": generate_plots
      if len(generate_plots)>1 else generate_plots[0],
    "display_plots": display_plots
      if len(display_plots)>1 else display_plots[0],
    "save_plots": save_plots if len(save_plots)>1 else save_plots[0],
    "eps": eps if len(eps)>1 else eps[0],
    "device": device if len(device)>1 else device[0],
    "rand_seed": rand_seed if len(rand_seed)>1 else rand_seed[0]}
  return output

def read_schedule(log_text):
  weights = [re.findall("\W(\w+)\W", weight_list)
    for weight_list in re.findall("weights\s+\[(.*)\]", log_text)]
  recon_mult = [float(val) for val in re.findall("recon_mult\s+(\d+\.?\d*)",
    log_text)]
  sparse_mult = [float(val)
    for val in re.findall("sparse_mult\s+(\d+\.?\d*)", log_text)]
  ent_mult = [float(val)
    for val in re.findall("ent_mult\s+(\d+\.?\d*)", log_text)]
  base_sup_mult = [float(val)
    for val in re.findall("base_sup_mult\s+(\d+\.?\d*)", log_text)]
  sup_mult = [float(val) for val in re.findall("[^_]sup_mult\s+(\d+\.?\d*)",
    log_text)]
  num_steps = [int(val) for val in re.findall("num_steps\s+(\d+)", log_text)]
  fb_mult = [float(val)
    for val in re.findall("fb_mult\s+(\d+\.?\d*)", log_text)]
  weight_lr = [[float(val) for val in lst]
    for lst in [re.findall("(\d+.?\d*)", res)
    for res in re.findall("weight_lr\s+\[(.*)\]", log_text)]]
  decay_steps = [[int(val) for val in lst]
    for lst in [re.findall("(\d+)", res)
    for res in re.findall("decay_steps\s+\[(.*)\]", log_text)]]
  decay_rate = [[float(val) for val in lst]
    for lst in [re.findall("(\d+\.?\d*)", res)
    for res in re.findall("decay_rate\s+\[(.*)\]", log_text)]]
  staircase = [[val=="True" for val in lst]
    for lst in [re.findall("(\w+)", res)
    for res in re.findall("staircase\s+\[(.*)\]", log_text)]]
  num_batches = [int(val)
    for val in re.findall("num_batches\s+(\d+)", log_text)]
  output = []
  for idx in range(len(weights)):
    output.append({
      "weights": weights[idx],
      "sparse_mult": sparse_mult[idx],
      "recon_mult": recon_mult[idx],
      "base_sup_mult": base_sup_mult[idx],
      "sup_mult": sup_mult[idx],
      "ent_mult": ent_mult[idx],
      "num_steps": num_steps[idx],
      "fb_mult": fb_mult[idx],
      "weight_lr": weight_lr[idx],
      "decay_steps": decay_steps[idx],
      "decay_rate": decay_rate[idx],
      "staircase": staircase[idx],
      "num_batches": num_batches[idx]})
  return output

"""
Generate dictionary of arrays that have loss values from log text
  Input: string containing log file text
  Output: dictionary containing arrays of loss values
"""
def read_loss(log_text):
  batch_index = np.array(
    [float(val) for val in re.findall("Global batch index is (\d+)", log_text)])
  perc_act = np.array(
    [float(val) for val in re.findall("percent active:\s+(\d+)", log_text)])
  a_max = np.array(
    [float(val) for val in re.findall("max val of a:\s+(\d+.?\d*)", log_text)])
  recon_quality = np.array(
    [float(val)
    for val in re.findall("recon pSNR dB:\s+(\-?\d+\.?\d*)", log_text)])
  euclidean_loss = np.array(
    [float(val)
    for val in re.findall("euclidean loss:\s+(\d+\.?\d*)", log_text)])
  sparse_loss = np.array(
    [float(val) for val in re.findall("sparse loss:\s+(\d+\.?\d*)", log_text)])
  unsupervised_loss = np.array(
    [float(val)
    for val in re.findall("unsupervised loss:\s+(\d+\.?\d*)", log_text)])
  supervised_loss = np.array(
    [float(val)
    for val in re.findall("[^un]supervised loss:\s+(\d+?\.?\d*)", log_text)])
  train_accuracy = np.array(
    [float(val)
    for val in re.findall("train accuracy:\s+(\d+\.?\d*)", log_text)])
  val_accuracy = np.array(
    [float(val)
    for val in re.findall("validation accuracy:\s+(\d+\.?\d*)", log_text)])
  assert batch_index.size != 0, "Global batch index was not found in input."
  output = {}
  output["batch_index"] = (batch_index
    if len(batch_index)>1 else batch_index[0])
  if recon_quality.size != 0:
    output["recon_quality"] = (recon_quality
      if len(recon_quality)>1 else recon_quality[0])
  else:
    output["recon_quality"] = [0 for _ in batch_index]
  if euclidean_loss.size != 0:
    output["euclidean_loss"] = (euclidean_loss
      if len(euclidean_loss)>1 else euclidean_loss[0])
  else:
    output["euclidean_loss"] = [0 for _ in batch_index]
  if perc_act.size != 0:
    output["perc_act"] = perc_act if len(perc_act)>1 else perc_act[0]
  else:
    output["perc_act"] = [0 for _ in batch_index]
  if sparse_loss.size != 0:
    output["sparse_loss"] = (sparse_loss
      if len(sparse_loss)>1 else sparse_loss[0])
  else:
    output["sparse_loss"] = [0 for _ in batch_index]
  if unsupervised_loss.size != 0:
    output["unsupervised_loss"] = (unsupervised_loss
      if len(unsupervised_loss)>1 else unsupervised_loss[0])
  else:
    output["unsupervised_loss"] = [0 for _ in batch_index]
  if supervised_loss.size != 0:
    output["supervised_loss"] = (supervised_loss
      if len(supervised_loss)>1 else supervised_loss[0])
  else:
    output["supervised_loss"] = [0 for _ in batch_index]
  if train_accuracy.size != 0:
    output["train_accuracy"] = (train_accuracy
      if len(train_accuracy)>1 else train_accuracy[0])
  else:
    output["train_accuracy"] = [0 for _ in batch_index]
  if val_accuracy.size != 0:
    output["val_accuracy"] = (val_accuracy
      if len(val_accuracy)>1 else val_accuracy[0])
  else:
    output["val_accuracy"] = [0 for _ in batch_index]
  return output
