## Copyright 2015 Yahoo Inc.
## Licensed under the terms of the New-BSD license. Please see LICENSE file in the project root for terms.
import os
import numpy as np
import matplotlib.pyplot as plt

"""
Normalize data
Outputs:
  data normalized so that when plotted 0 will be midlevel grey
Args:
  data: np.ndarray
"""
def normalize_data(data):
  norm_data = data.squeeze()
  if np.max(np.abs(data)) > 0:
    norm_data = (data / np.max(np.abs(data))).squeeze()
  return norm_data

"""
Pad data with ones for visualization
Outputs:
  padded version of input
Args:
  data: np.ndarray
"""
def pad_data(data):
  n = int(np.ceil(np.sqrt(data.shape[0])))
  padding = (((0, n ** 2 - data.shape[0]),
    (1, 1), (1, 1))                       # add some space between filters
    + ((0, 0),) * (data.ndim - 3))        # don't pad the last dimension (if there is one)
  padded_data = np.pad(data, padding, mode="constant", constant_values=1)
  # tile the filters into an image
  padded_data = padded_data.reshape((n, n) + padded_data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, padded_data.ndim + 1)))
  padded_data = padded_data.reshape((n * padded_data.shape[1], n * padded_data.shape[3]) + padded_data.shape[4:])
  return padded_data

"""
TODO: Generate a plot of top & bottom phi for a given digit label
e.g. for data that has shape [10, 12, 28, 28]:
|--------------------------------|
| Class    Top         Bot       |
|   0      [] [] []    [] [] []  |
|          [] [] []    [] [] []  |
|                                |
|   1      [] [] []    [] [] []  |
|          [] [] []    [] [] []  |
|   ...    ...         ...       |
|                                |
|   9      [] [] []    [] [] []  |
|          [] [] []    [] [] []  |
|--------------------------------|
Args:
  data: numpy ndarray of shape (Classes, Num_phi, Width, Height)
  title: string for title of figure
  save_filename: string holding output directory for writing,
    figures will not display with GUI if set
"""
#def save_select_phi(data, title="", save_filename=""):
  
"""
Generate a bar graph of data
Args:
  data: numpy ndarray of shape (N,)
  xticklabels: [list of str] indicating the labels for the xticks
  out_filename: [str] indicating where the file should be saved
  xlabel: [str] indicating the x-axis label
  ylabel: [str] indicating the y-axis label
  title: [str] indicating the plot title
"""
def save_bar(data, xticklabels=None, out_filename="./bar_fig.pdf", xlabel="",
  ylabel="", title=""):
  if not xticklabels:
    xticklabels = [str(val) for val in np.arange(len(data))]
  fig, axis = plt.subplots(1)
  bar = axis.bar(np.arange(len(data)), data)
  axis.set_xlabel(xlabel)
  axis.set_ylabel(ylabel)
  axis.set_xticklabels(xticklabels)
  fig.suptitle(title, y=1.0, x=0.5)
  fig.savefig(out_filename, transparent=True)
  plt.close(fig)

"""
Plot of model neuron inputs over time
Args:
  data: [dict] with each trace, with keys [b, u, a, ga, fb, images]
    Dictionary is created by analyze_lca.compute_inference()
  base_filename: [str] containing the base filename for outputs. Since multiple
    outputs are created, file extension should not be included.
  file_ext: [str] containing the file extension for the output images
"""
def save_inference_traces(data, base_filename, file_ext):
  (num_images, num_timesteps, num_neurons) = data["b"].shape
  img_idx = 0
  max_val = np.max([data["b"][img_idx,...], data["u"][img_idx,...],
    data["ga"][img_idx,...], data["a"][img_idx,...]])
  min_val = np.min([data["b"][img_idx,...], data["u"][img_idx,...],
    data["ga"][img_idx,...], data["a"][img_idx,...]])
  fig, sub_axes = plt.subplots(int(np.sqrt(num_neurons)+1),
    int(np.sqrt(num_neurons)))
  for (axis_idx, axis) in enumerate(fig.axes):
    if axis_idx < num_neurons:
      t = np.arange(data["b"].shape[1])
      b = data["b"][img_idx,:,axis_ix]
      u = data["u"][img_idx,:,axis_ix]
      ga = data["ga"][img_idx,:,axis_ix]
      fb = data["fb"][img_idx,:,axis_ix]
      a = data["a"][img_idx,:,axis_ix]
      axis.plot(t, b, linewidth=0.25, color="g", label="b")
      axis.plot(t, u, linewidth=0.25, color="b", label="u")
      axis.plot(t, ga, linewidth=0.25, color="r", label="Ga")
      axis.plot(t, fb, linewidth=0.25, color="y", label="fb")
      axis.plot(t, [0 for _ in t], linewidth=0.25,
        color="k", linestyle="-", label="zero")
    #axis.set_ylim((min_val, max_val))
    #axis.set_aspect(1.0001)
    axis.tick_params(
      axis="both",
      which="both",
      bottom="off",
      top="off",
      left="off",
      right="off",
      labelbottom="off",
      labeltop="off",
      labelleft="off",
      labelright="off")
    if a.any() > 0:
      for spine in axis.spines.values():
        spine.set_edgecolor('magenta')
  num_pixels = np.size(data["images"][img_idx])
  image = data["images"][img_idx,...].reshape(int(np.sqrt(num_pixels)),
    int(np.sqrt(num_pixels)))
  sub_axes[int(np.sqrt(num_neurons)), 0].imshow(image, cmap="Greys",
    interpolation="nearest")
  for plot_col in range(int(np.sqrt(num_neurons))):
    sub_axes[int(np.sqrt(num_neurons)), plot_col].axis("off")
    sub_axes[int(np.sqrt(num_neurons)), plot_col].get_xaxis().set_visible(False)
    sub_axes[int(np.sqrt(num_neurons)), plot_col].get_yaxis().set_visible(False)
    sub_axes[int(np.sqrt(num_neurons)), plot_col].tick_params(
      axis="both",
      which="both",
      bottom="off",
      top="off",
      left="off",
      right="off",
      labelbottom="off",
      labeltop="off",
      labelleft="off",
      labelright="off")
  #sub_axes[0, 19].legend(bbox_to_anchor=(1.1, 1.0), ncol=1, fancybox=True)

  #sub_axes[0].set_xlabel("Time Step (dt = 1 msec)")
  #sub_axes[2].set_ylabel("LCA Input Traces")
  #sub_axes[2].yaxis.set_label_coords(ylabel_xpos, 0.5)
  fig.suptitle("LCA Activity", y=0.99, x=0.5)
  out_filename = (base_filename+"_lca_traces"+file_ext)
  fig.savefig(out_filename, transparent=True)
  plt.close(fig)

"""
Plot of inference statistics, including reconstruction, sparsity, cross-entropy
Args:
  data: [dict] with keys [unsup_loss, euc_loss, psnr, recon, images]
    Dictionary is created by analyze_lca.compute_inference()
  base_filename: [str] containing the base filename for outputs. Since multiple
    outputs are created, file extension should not be included.
  file_ext: [str] containing the file extension for the output images
  num_skip: [int] specifying how many inference time steps to skip when
    creating plots. Default is to print all time steps.
"""
def save_inference_stats(data, base_filename, file_ext, num_skip=1):
  ## Loss over time
  fig, sub_axes = plt.subplots(2,3)
  unsup_loss = data["unsup_loss"]
  (nimgs, nsteps) = unsup_loss.shape
  t = np.arange(nsteps)
  unsup_loss_mean = np.mean(unsup_loss, axis=0)
  unsup_loss_sem = np.std(unsup_loss, axis=0) / np.sqrt(nimgs)
  sub_axes[0,0].plot(t, unsup_loss_mean, "k-")
  sub_axes[0,0].fill_between(t, unsup_loss_mean-unsup_loss_sem,
    unsup_loss_mean+unsup_loss_sem, alpha=0.5)

  euc_loss = data["euc_loss"]
  (nimgs, nsteps) = euc_loss.shape
  t = np.arange(nsteps)
  euc_mean = np.mean(euc_loss, axis=0)
  euc_sem = np.std(euc_loss, axis=0) / np.sqrt(nimgs)
  sub_axes[0,1].plot(t, euc_mean, "k-")
  sub_axes[0,1].fill_between(t, euc_mean-euc_sem,
    euc_mean+euc_sem, alpha=0.5)

  sparse_loss = data["sparse_loss"]
  (nimgs, nsteps) = euc_loss.shape
  t = np.arange(nsteps)
  euc_mean = np.mean(euc_loss, axis=0)
  euc_sem = np.std(euc_loss, axis=0) / np.sqrt(nimgs)
  sub_axes[0,2].plot(t, euc_mean, "k-")
  sub_axes[0,2].fill_between(t, euc_mean-euc_sem,
    euc_mean+euc_sem, alpha=0.5)

  xent_loss = data["xent_loss"]
  (nimgs, nsteps) = euc_loss.shape
  t = np.arange(nsteps)
  euc_mean = np.mean(euc_loss, axis=0)
  euc_sem = np.std(euc_loss, axis=0) / np.sqrt(nimgs)
  sub_axes[1,0].plot(t, euc_mean, "k-")
  sub_axes[1,0].fill_between(t, euc_mean-euc_sem,
    euc_mean+euc_sem, alpha=0.5)

  psnr = data["psnr"]
  (nimgs, nsteps) = psnr.shape
  t = np.arange(nsteps)
  psnr_mean = np.mean(psnr, axis=0)
  psnr_sem = np.std(psnr, axis=0) / np.sqrt(nimgs)
  sub_axes[1,1].plot(t, psnr_mean, "k-")
  sub_axes[1,1].fill_between(t, psnr_mean-psnr_sem,
    psnr_mean+psnr_sem, alpha=0.5)

  sub_axes[0,0].get_xaxis().set_ticklabels([])
  sub_axes[0,1].get_xaxis().set_ticklabels([])
  sub_axes[0,2].set_xlabel("Time Step (dt = 1 msec)")
  sub_axes[1,0].get_xaxis().set_ticklabels([])
  sub_axes[1,1].set_xlabel("Time Step (dt = 1 msec)")
  sub_axes[1,2].get_xaxis().set_ticklabels([])

  sub_axes[0,0].set_ylabel("Unsupervised Loss")
  sub_axes[0,1].set_ylabel("Euclidean Loss")
  sub_axes[0,2].set_ylabel("Sparse Loss")
  sub_axes[1,0].set_ylabel("Cross Entropy Loss")
  sub_axes[1,1].set_ylabel("Recon pSNR dB")
  sub_axes[1,2].get_yaxis().set_ticklabels([])

  ylabel_xpos = -0.1
  sub_axes[0,0].yaxis.set_label_coords(ylabel_xpos, 0.5)
  sub_axes[0,1].yaxis.set_label_coords(ylabel_xpos, 0.5)
  sub_axes[0,2].yaxis.set_label_coords(ylabel_xpos, 0.5)
  sub_axes[1,0].yaxis.set_label_coords(ylabel_xpos, 0.5)
  sub_axes[1,1].yaxis.set_label_coords(ylabel_xpos, 0.5)
  fig.suptitle("Average Statistics During Inference", y=1.0, x=0.5)
  out_filename = (base_filename+"_inference_stats"+file_ext)
  fig.savefig(out_filename, transparent=True)
  plt.close(fig)

  ## Recon error images
  (num_images, num_timesteps, num_pixels) = data["recon"].shape
  title = "Input"
  out_filename = (base_filename+"_input"+file_ext)
  save_data_tiled(data["images"].reshape(num_images, int(np.sqrt(num_pixels)),
    int(np.sqrt(num_pixels))), normalize=False, title=title,
    save_filename=out_filename)

  for timestep in range(0, num_timesteps, num_skip):
    error = data["images"] - data["recon"][:, timestep, :]
    title = "Input - Reconstruction for timestep "+str(timestep)
    out_filename = (base_filename+"_recon_err_t"+str(timestep)+file_ext)
    save_data_tiled(error.reshape(num_images, int(np.sqrt(num_pixels)),
      int(np.sqrt(num_pixels))), normalize=False, title=title,
      save_filename=out_filename)

    title = "Reconstruction for timestep "+str(timestep)
    out_filename = (base_filename+"_recon_t"+str(timestep)+file_ext)
    save_data_tiled(data["recon"][:, timestep, :].reshape(num_images,
      int(np.sqrt(num_pixels)), int(np.sqrt(num_pixels))), normalize=False,
      title=title, save_filename=out_filename)

"""
Generate cross-entropy, euclidean, and sparse loss vals
Outputs:
  fig: [int] corresponding to the figure number
Args:
  data: [dict] containing keys
    "batch_index", "euclidean_loss", "sparse_loss", "supervised_loss"
    all dict elements should contain lists of equal length
  out_filename: [str] containing the complete output filename
"""
def save_losses(data, out_filename):
  fig, sub_axes = plt.subplots(4)
  axis_image = [None]*4
  if "euclidean_loss" in data.keys():
    axis_image[0] = sub_axes[0].plot(data["batch_index"], data["euclidean_loss"])
  if "sparse_loss" in data.keys():
    axis_image[1] = sub_axes[1].plot(data["batch_index"], data["sparse_loss"])
  if "supervised_loss" in data.keys():
    axis_image[2] = sub_axes[2].plot(data["batch_index"], data["supervised_loss"])
    if "unsupervised_loss" in data.keys():
      axis_image[3] = sub_axes[3].plot(data["batch_index"],
        data["unsupervised_loss"]+data["supervised_loss"])
  sub_axes[0].get_xaxis().set_ticklabels([])
  sub_axes[1].get_xaxis().set_ticklabels([])
  sub_axes[2].get_xaxis().set_ticklabels([])
  sub_axes[0].locator_params(axis="y", nbins=5)
  sub_axes[1].locator_params(axis="y", nbins=5)
  sub_axes[2].locator_params(axis="y", nbins=5)
  sub_axes[3].locator_params(axis="y", nbins=5)
  sub_axes[3].set_xlabel("Batch Number")
  sub_axes[0].set_ylabel("Euclidean")
  sub_axes[1].set_ylabel("Sparse")
  sub_axes[2].set_ylabel("Cross Entropy")
  sub_axes[3].set_ylabel("Total")
  ylabel_xpos = -0.1
  sub_axes[0].yaxis.set_label_coords(ylabel_xpos, 0.5)
  sub_axes[1].yaxis.set_label_coords(ylabel_xpos, 0.5)
  sub_axes[2].yaxis.set_label_coords(ylabel_xpos, 0.5)
  sub_axes[3].yaxis.set_label_coords(ylabel_xpos, 0.5)
  fig.suptitle("Average Losses per Batch", y=1.0, x=0.5)
  fig.savefig(out_filename, transparent=True)
  plt.close(fig)

"""
Generate a plot of the performance, as calculated 
in analyze_lca.compute_performance()
Outputs:
  fig: list containing figure numbers for generated figures
Args:
  data: [list of lists of np.ndarrays] containing the data to be plotted.
    The first index is for test conditions, which are identified by "labels"
    input. The second index specifies the information to be plotted, which is
    identified by the "plot_names" variable. len(plot_names) plots will be
    created.
  labels: [list of ints] that contain the number of labeled images used
    in the test case.
  out_filename: [str] containing the output filename
  run_type: [str] containing the name of the type of run (e.g. "test" or "val")
  plot_sem: [bool] If set, standard error of the mean will also be plotted as
    error-bars.
"""
def save_performance(data, labels, out_filename="./out.pdf", run_type="",
    plot_sem=False):
    plot_names = [run_type+"_acc", run_type+"_cross_ent",
      run_type+"_sparsity", run_type+"_recon_pSNRdB"]
    fig = [None]*len(plot_names)
    for plot_idx in range(len(plot_names)):
      fig[plot_idx] = plt.figure()
      ax0 = fig[plot_idx].add_subplot(1,1,1)
      bar_width = 0.35
      num_imgs = len(data[0][plot_idx])
      mean_vals = [np.mean(data[condition_idx][plot_idx])
        for condition_idx in range(len(labels))]
      x_vals = np.arange(len(mean_vals))
      if plot_sem:
        sem_vals = [np.std(data[condition_idx][plot_idx])/np.sqrt(num_imgs)
          for condition_idx in range(len(labels))]
        rects = ax0.bar(x_vals, mean_vals, yerr=sem_vals)
      else:
        rects = ax0.bar(x_vals, mean_vals)
      ax0.set_ylabel(plot_names[plot_idx])
      ax0.set_title("Average Performance on "+run_type+" Set")
      ax0.set_xticks(x_vals+bar_width)
      ax0.set_xticklabels([str(label) for label in labels])
      ax0.set_xlabel("Number of Labeled Examples in Training Set")
      out_file, out_ext = os.path.splitext(out_filename)
      fig[plot_idx].savefig((out_file+"_"+plot_names[plot_idx]+out_ext),
        transparent=True)
      plt.close(fig[plot_idx])
    return fig

"""
Generate training and validation accuracy
Outputs:
  fig: [int] corresponding to the figure number
Args:
  data: [dict] containing keys
    "batch_index", "train_accuracy", "val_accuracy"
    data["batch_index"] and data["train_accuracy"] should contain lists
    of equal length.
  out_filename: [str] containing the complete output filename

TODO: The x values for the validation plot are approximate.
      It would be better to grab the exact global batch index
      for the corresponding validation accuracy value. This
      would be done in utils/parse_logfile.py
"""
def save_accuracy(data, out_filename):
  fig, sub_axes = plt.subplots(2)
  ylabel_xpos = -0.1
  axis_image = [None]*4
  axis_image[0] = sub_axes[0].scatter(data["batch_index"],
    data["train_accuracy"], c="b", marker=".", s=5.0)
  z = np.polyfit(data["batch_index"], data["train_accuracy"], deg=1)
  p = np.poly1d(z)
  axis_image[1] = sub_axes[0].plot(data["batch_index"],
    p(data["batch_index"]), "r--")
  sub_axes[0].set_ylabel("Train Accuracy")
  sub_axes[0].set_ylim((0, 1.0))
  sub_axes[0].set_xlim((0, data["batch_index"][-1]))
  sub_axes[0].yaxis.set_label_coords(ylabel_xpos, 0.5)
  if "val_accuracy" in data.keys():
    val_xdat = np.linspace(data["batch_index"][0], data["batch_index"][-1],
      len(data["val_accuracy"]))
    axis_image[2] = sub_axes[1].scatter(val_xdat, data["val_accuracy"],
      c="b", marker=".", s=5.0)
    z = np.polyfit(val_xdat, data["val_accuracy"], deg=1)
    p = np.poly1d(z)
    axis_image[3] = sub_axes[1].plot(val_xdat, p(val_xdat), "r--")
    sub_axes[1].set_xlabel("Batch Number")
    sub_axes[1].set_ylabel("Val Accuracy")
    sub_axes[1].set_xlim((0, data["batch_index"][-1]))
    sub_axes[1].set_ylim((0, 1.0))
    sub_axes[1].yaxis.set_label_coords(ylabel_xpos, 0.5)
  fig.suptitle("Average Accuracy per Batch", y=1.0, x=0.5)
  fig.savefig(out_filename, transparent=True)
  plt.close(fig)

"""
Generate reconstruction quality
Outputs:
  fig: [int] corresponding to the figure number
Args:
  data: [dict] containing keys
    "batch_index", "recon_quality"
    all dict elements should contain lists of equal length
  out_filename: [str] containing the complete output filename
"""
def save_recon(data, out_filename):
  fig = plt.figure()
  ax0 = fig.add_subplot(1,1,1)
  l0 = ax0.plot(data["batch_index"], data["recon_quality"])
  ax0.set_xlabel("Batch Number")
  ax0.set_ylabel("Reconstruction Quality (pSNR dB)")
  ax0.set_title("Average Reconstruction Quality per Batch")
  fig.savefig(out_filename, transparent=True)
  plt.close(fig)

"""
Save figure for input data as a tiled image
Outputs:
  fig: index for figure call
  sub_axis: index for subplot call
  axis_image: index for imshow call
Inpus:
  data: [np.ndarray] of shape:
    (height, width) - single image
    (n, height, width) - n images tiled with dim (sqrt(n), sqrt(n))
  normalize: [bool] indicating whether the data should be streched (normalized)
    This is recommended for dictionary plotting.
  title: [str] for title of figure
  save_filename: [str] holding output directory for writing,
    figures will not display with GUI if set
"""
def save_data_tiled(data, normalize=False, title="", save_filename=""):
  if normalize:
    data = normalize_data(data)
  if len(data.shape) >= 3:
    data = pad_data(data)
  fig, sub_axis = plt.subplots(1)
  axis_image = sub_axis.imshow(data, cmap="Greys", interpolation="nearest")
  axis_image.set_clim(vmin=-1.0, vmax=1.0)
  cbar = fig.colorbar(axis_image)
  sub_axis.tick_params(
   axis="both",
   bottom="off",
   top="off",
   left="off",
   right="off")
  sub_axis.get_xaxis().set_visible(False)
  sub_axis.get_yaxis().set_visible(False)
  sub_axis.set_title(title)
  if save_filename == "":
    save_filename = "./output.ps"
  fig.savefig(save_filename, transparent=True, bbox_inches="tight", pad_inches=0.01)
  plt.close(fig)

"""
Save input data as an image without reshaping
Outputs:
  fig: index for figure call
  sub_axis: index for subplot call
  axis_image: index for imshow call
Inpus:
  data: np.ndarray of shape (height, width) or (n, height, width)
  title: string for title of figure
  save_filename: string holding output directory for writing,
    figures will not display with GUI if set
"""
def save_data(data, title="", save_filename=""):
  fig, sub_axis = plt.subplots(1)
  axis_image = sub_axis.imshow(data, cmap="Greys", interpolation="nearest")
  cbar = fig.colorbar(axis_image)
  sub_axis.tick_params(
   axis="both",
   bottom="off",
   top="off",
   left="off",
   right="off")
  fig.suptitle(title, y=1.05)
  if save_filename == "":
    save_filename = "./output.ps"
  fig.savefig(save_filename, transparent=True, bbox_inches="tight", pad_inches=0.01)
  plt.close(fig)

"""
Display input data as an image with reshaping
Outputs:
  fig: index for figure call
  sub_axis: index for subplot call
  axis_image: index for imshow call
Inpus:
  data: np.ndarray of shape (height, width) or (n, height, width)
  normalize: [bool] indicating whether the data should be streched (normalized)
    This is recommended for dictionary plotting.
  title: string for title of figure
  prev_fig: tuple containing (fig, sub_axis, axis_image) from previous
    display_data() call
  TODO: Allow for color weight vis
"""
def display_data_tiled(data, normalize=False, title="", prev_fig=None):
  if normalize:
    data = normalize_data(data)
  if len(data.shape) >= 3:
    data = pad_data(data)
  if prev_fig is None:
    fig, sub_axis = plt.subplots(1)
    axis_image = sub_axis.imshow(data, cmap="Greys", interpolation="nearest")
    axis_image.set_clim(vmin=-1.0, vmax=1.0)
    cbar = fig.colorbar(axis_image)
    sub_axis.tick_params(
     axis="both",
     bottom="off",
     top="off",
     left="off",
     right="off")
  else:
    (fig, sub_axis, axis_image) = prev_fig
    axis_image.set_data(data)
  fig.suptitle(title, y=1.05)
  if prev_fig is None:
    fig.show()
  else:
    fig.canvas.draw()
  return (fig, sub_axis, axis_image)

"""
Display input data as an image without reshaping
Outputs:
  fig: index for figure call
  sub_axis: index for subplot call
  axis_image: index for imshow call
Inpus:
  data: np.ndarray of shape (height, width) or (n, height, width) 
  title: string for title of figure
  prev_fig: tuple containing (fig, sub_axis, axis_image) from previous display_data() call
"""
def display_data(data, title="", prev_fig=None):
  if prev_fig is None:
    fig, sub_axis = plt.subplots(1)
    axis_image = sub_axis.imshow(data, cmap="Greys", interpolation="nearest")
    cbar = fig.colorbar(axis_image)
    sub_axis.tick_params(
     axis="both",
     bottom="off",
     top="off",
     left="off",
     right="off")
  else:
    (fig, sub_axis, axis_image) = prev_fig
    axis_image.set_data(data)
    axis_image.autoscale()
  fig.suptitle(title, y=1.05)
  if prev_fig is None:
    fig.show()
  else:
    fig.canvas.draw()
  return (fig, sub_axis, axis_image)
