# SemiSupervisedLCA
Modified LCA model to include feedback

### TODO - New code
 * Add manually computed Entropy gradient to update\_u equation
 * Plot incorrectly labeled examples from val or test set using save\_data\_tiled()
 * Histogram y\_ values to determine relative scales
 * Write in normalization (regularization) method for classification 'w' matrix
 * Add alternative regularizers - leaky learning & dropout on 'phi' or 'w'

### TODO - Code fixes
 * Fix plotting so that num\_neurons does not have to have a sqrt
 * Fix auto-differentiation for LCAF inference

Copyright 2015 Yahoo Inc.
Licensed under the terms of the New-BSD license.
Please see LICENSE file in the project root for terms.
