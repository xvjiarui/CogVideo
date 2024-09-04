# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

from sat.helpers import print_rank0


class AnnealingLR(_LRScheduler):
    """Anneals the learning rate from start to zero along a cosine curve."""

    DECAY_STYLES = ['linear', 'cosine', 'constant']

    def __init__(self, optimizer, start_lr, warmup_iter, num_iters, decay_style=None, last_iter=-1, decay_ratio=0.5):
        assert warmup_iter <= num_iters
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.warmup_iter = warmup_iter
        self.num_iters = last_iter + 1
        self.end_iter = num_iters
        self.decay_style = decay_style.lower() if isinstance(decay_style, str) else None
        self.decay_ratio = decay_ratio
        self.step(self.num_iters)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print_rank0(f'learning rate decaying style {self.decay_style}, ratio {self.decay_ratio}')
        assert self.decay_style in self.DECAY_STYLES

    def get_lr_scale(self):
        if self.warmup_iter > 0 and self.num_iters <= self.warmup_iter:
            return self.num_iters / self.warmup_iter
        else:
            if self.decay_style == 'linear':
                return (self.end_iter-(self.num_iters-self.warmup_iter))/self.end_iter
            elif self.decay_style == 'cosine':
                decay_step_ratio = min(1.0, (self.num_iters - self.warmup_iter) / self.end_iter)
                cosine_decay = (math.cos(math.pi * decay_step_ratio) + 1) / 2
                return cosine_decay * (1 - self.decay_ratio) + self.decay_ratio
            elif self.decay_style == 'constant':
                return 1.0
            else:
                raise ValueError(f'Unknown decay style: {self.decay_style}')

    def step(self, step_num=None):
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        lr_scale = self.get_lr_scale()
        for group in self.optimizer.param_groups:
            lr = self.start_lr * lr_scale
            if isinstance(lr, torch.Tensor):
                lr_val = lr.item() if isinstance(lr, torch.Tensor) else lr  # type: ignore[attr-defined]
                group["lr"].fill_(lr_val)
            else:
                group["lr"] = lr

    def state_dict(self):
        sd = {
                'warmup_iter': self.warmup_iter,
                'num_iters': self.num_iters,
                'decay_style': self.decay_style,
                'end_iter': self.end_iter,
                'decay_ratio': self.decay_ratio
        }
        return sd

    def load_state_dict(self, sd):
        pass # disable this 
