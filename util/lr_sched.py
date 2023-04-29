# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

def adjust_learning_rate(optimizer, epoch):
   
  init_lr = 1e-1
  
  """Decay the learning rate with half-cycle cosine after warmup"""
  if epoch <= 20:
      lr = init_lr / 10

  for param_group in optimizer.param_groups:
      if "lr_scale" in param_group:
          param_group["lr"] = lr * param_group["lr_scale"]
      else:
          param_group["lr"] = lr
  return lr

