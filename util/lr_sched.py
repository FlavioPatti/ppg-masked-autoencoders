# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

warmup_epochs = 5
epochs = 50
min_lr = 0
blr = 1e-3

def adjust_learning_rate(optimizer, epoch):
    """Decay the learning rate with half-cycle cosine after warmup"""
    #dopo warmup_epochs lr raggiunge il valore di blr
    if epoch < warmup_epochs:
        lr = blr * epoch / warmup_epochs 
        #print(f"lr1 = {lr}")
    else:
        lr = min_lr + (blr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
        #print(f"lr2 = {lr}")
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
