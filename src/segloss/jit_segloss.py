import os
import logging
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
seg_loss_cuda_jit = load(
    'seg_loss_cuda_jit', ['./segloss/src/seg_loss_torch_api.cc', './segloss/src/seg_loss_kernel.cu', './segloss/src/logsumexp_custom.cu', './segloss/src/reduce_kernel_v3.cu', './segloss/src/cuda_error_checking.cu'], verbose=True)
from seg_loss_cuda_jit import SegLossTorch
from apex import amp

class SegLossFunction(torch.autograd.Function):

    @staticmethod
    @amp.float_function
    def forward(ctx, segLoss, W, label_sizes, prob_sizes, label_batch,
                is_sum):
        loss = W.new_zeros(1)
        B, T, S, V, L = W.size(0), W.size(1), W.size(2), W.size(3), label_sizes.max().item()
        alpha_1d = W.new_zeros((T+1)*B)
        alpha_2d = W.new_zeros((T+1)*B*(L+1))
        beta_1d = W.new_zeros((T+1)*B)
        beta_2d = W.new_zeros((T+1)*B*(L+1))
        loss = segLoss.cuda_forward(W, loss, label_sizes, prob_sizes, label_batch,
                                    alpha_1d, alpha_2d, beta_1d, beta_2d,
                                    is_sum)
        ctx.is_sum = is_sum
        ctx.segLoss = segLoss
        ctx.save_for_backward(W, label_sizes, prob_sizes, label_batch,
                              alpha_1d, alpha_2d, beta_1d, beta_2d)
        return loss

    @staticmethod
    @amp.float_function
    def backward(ctx, grad_output):
        W, label_sizes, prob_sizes, label_batch, \
            alpha_1d, alpha_2d, beta_1d, beta_2d = ctx.saved_tensors
        segLoss, is_sum = ctx.segLoss, ctx.is_sum
        grad_W = segLoss.cuda_backward(W, grad_output,
                                       label_sizes, prob_sizes, label_batch,
                                       alpha_1d, alpha_2d, beta_1d, beta_2d, is_sum)
        return None, grad_W, None, None, None, None


class SegLossModule(nn.Module):
    def __init__(self, chunk_size, verbose=True):
        super(SegLossModule, self).__init__()
        self.segLoss = SegLossTorch(chunk_size, verbose)

    def forward(self, W, label_batch, label_sizes, prob_sizes, reduction):
        if reduction == 'SUM':
            is_sum = True
        elif reduction == 'MEAN':
            is_sum = False
        else:
            raise NotImplementedError("Option for reduction: SUM|MEAN")
        return SegLossFunction.apply(self.segLoss, W, label_sizes, prob_sizes, label_batch, is_sum)
