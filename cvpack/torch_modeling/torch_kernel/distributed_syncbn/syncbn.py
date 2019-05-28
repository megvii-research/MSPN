import math
from queue import Queue
from IPython import embed

import torch
import torch.distributed as dist
import torch.cuda.comm as comm
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F

import syncbn_gpu


class DistributedSyncBNFucntion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, running_mean, running_var,
            training=True, momentum=0.1, eps=1e-5, sync=True):
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.sync = sync

        if ctx.training:
            ex, exs = syncbn_gpu.batch_norm_collect_statistics(x)

            if ctx.sync:
                world_size = dist.get_world_size()
                ex_all  = torch.empty(world_size, ex.size(0), dtype=ex.dtype, device=ex.device)
                exs_all = torch.empty(world_size, ex.size(0), dtype=ex.dtype, device=ex.device)
                ex_l    = [ex_all.narrow(0, i, 1) for i in range(world_size)]
                exs_l   = [exs_all.narrow(0, i, 1) for i in range(world_size)]
                dist.all_gather(ex_l, ex)
                dist.all_gather(exs_l, exs)

                ex = ex_all.mean(0)
                exs = exs_all.mean(0)

            var = exs - ex ** 2
            running_mean.mul_(1 - ctx.momentum).add_(ctx.momentum * ex)
            running_var.mul_(1 - ctx.momentum).add_(ctx.momentum * var)

            ctx.mark_dirty(running_mean, running_var)

            y = syncbn_gpu.batch_norm_transform_input(x, gamma, beta, ex, exs, ctx.eps)

            ctx.save_for_backward(x, ex, exs, gamma, beta)

        return y

    @staticmethod
    def backward(ctx, grad_ouput):
        x, ex, exs, gamma, beta = ctx.saved_tensors

        grad_gamma, grad_beta, grad_ex, grad_exs = \
                syncbn_gpu.batch_norm_collect_grad_statistics(x, grad_ouput, gamma, ex, exs, ctx.eps)

        if ctx.training:
            if ctx.sync:
                world_size = dist.get_world_size()
                grad_ex_all  = torch.empty(world_size, grad_ex.size(0), dtype=grad_ex.dtype, device=grad_ex.device)
                grad_exs_all = torch.empty(world_size, grad_ex.size(0), dtype=grad_ex.dtype, device=grad_ex.device)
                grad_ex_l    = [grad_ex_all.narrow(0, i, 1) for i in range(world_size)]
                grad_exs_l   = [grad_exs_all.narrow(0, i, 1) for i in range(world_size)]
                dist.all_gather(grad_ex_l, grad_ex)
                dist.all_gather(grad_exs_l, grad_exs)

                grad_ex = grad_ex_all.mean(0)
                grad_exs = grad_exs_all.mean(0)

        grad_input = syncbn_gpu.batch_norm_input_backward(x, grad_ouput, gamma, ex, exs, grad_ex, grad_exs, ctx.eps)

        return grad_input, grad_gamma, grad_beta, None, None, None, None, None, None


class DistributedSyncBN(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, sync=True):
        super(DistributedSyncBN, self).__init__(num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True)
        self.sync = sync

    def forward(self, x):
        if self.training and self.sync:
            return DistributedSyncBNFucntion.apply(x, self.weight, self.bias, self.running_mean, self.running_var,
                self.training, self.momentum, self.eps, self.sync)
        else:
            exponential_average_factor = 0.0

            if self.training and self.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                    if self.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.momentum

            return F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)


if __name__ == '__main__':
    pass
