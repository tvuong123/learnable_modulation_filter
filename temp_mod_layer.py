import scipy.signal as signal
import torch
from torch import nn
import torch.nn.functional as F
import math
from pdb import set_trace
from torch import Tensor
import numpy as np


class GaborConv1D(nn.Module):
   

    def __init__(self, filters, kernel_size, init_cf, init_bw, use_bias,
                 name, trainable, sample_rate=100.0, norm=True, transpose_input=False, strf_type='real', train_window=False, single_cf=False):

        super(GaborConv1D, self).__init__()
        self._filters = filters
        self._kernel_size = int(sample_rate * kernel_size) if int(
            sample_rate * kernel_size) % 2 == 1 else int(sample_rate * kernel_size) + 1
        self._padding = int(self._kernel_size//2)
        self._use_bias = use_bias
        self._sample_rate = sample_rate
        self.init_cf = init_cf
        self.init_bw = init_bw
        self.trainable = trainable
        self.transpose_input = transpose_input
        self.strf_type = strf_type
        self.train_window = train_window
        self.single_cf = single_cf

        if not self.single_cf:
            initialized_cf = torch.ones(self._filters)
        else:
            initialized_cf = torch.ones(1, 1)
        initialized_cf *= init_cf / (sample_rate / 2.0) * math.pi
        self.cf_ = nn.Parameter(
            initialized_cf, requires_grad=trainable)
        self.norm = norm
        window_function = torch.FloatTensor(signal.firwin(
            int(self._kernel_size), init_bw/sample_rate*2, window='hamming')).unsqueeze(0)
        self.lpf = torch.ones(len(self.cf_), len(window_function[0]))
        self.lpf = nn.Parameter(
            window_function * self.lpf, requires_grad=train_window)

        if self.norm:
            self.instance = nn.InstanceNorm1d(
                self._filters, affine=True, track_running_stats=False)
          
        if self._use_bias:
            # TODO: validate that requires grad is the same as trainable
            self._bias = nn.Parameter(torch.zeros(
                self.filters*2,), requires_grad=trainable)

        self.register_buffer("gabor_filter_init_t",
                             torch.arange(-(self._kernel_size // 2), (self._kernel_size + 1) // 2, dtype=torch.float32))

    def gabor_impulse_response(self, t: Tensor, center: Tensor,
                               ) -> Tensor:
        """Computes the gabor impulse response."""
        if not self.single_cf:
            cf = center
        else:
            cf = center.repeat(self._filters, 1)

        inside = torch.tensordot(cf, t, dims=0)
        return torch.cos(inside) * self.lpf, torch.sin(inside) * self.lpf

    def get_gabor_filters(self):
        filters = self.gabor_impulse_response(
            self.gabor_filter_init_t, self.cf_.clamp(0, math.pi))

        return filters

    def forward(self, x):

        real_filters, img_filters = self.get_gabor_filters()
        if not self.single_cf:
            real_filters = real_filters.unsqueeze(1)
            img_filters = img_filters.unsqueeze(1)

        if self.transpose_input:
            x = x.transpose(2, 1)
       
        if self.strf_type == 'real':
            out = F.conv1d(x, real_filters, bias=self._bias if self._use_bias else None,
                           stride=1, padding=self._padding, groups=self._filters)
        elif self.strf_type == 'both':

            output1, output2 = F.conv1d(x, real_filters,
                                        bias=self._bias if self._use_bias else None, stride=1,
                                        padding=self._padding, groups=self._filters), F.conv1d(x, img_filters,
                                                                                               bias=self._bias if self._use_bias else None, stride=1,
                                                                                               padding=self._padding, groups=self._filters)
            out = torch.cat(
                (output1.unsqueeze(-1), output2.unsqueeze(-1)), dim=-1)

        if self.transpose_input:
            out = out.transpose(2, 1)

        return out

    def __repr__(self):
        """GaborConv1d"""
        report = """
            +++++ Gabor Filter Kernels [{}], kernel_size [{}] sample_rate [{}], init_cf [{}], init_bw [{}] trainable [{}] strf type [{}]  train window [{}] single cf [{}]+++++

        """.format(self._filters, self._kernel_size, self._sample_rate, self.init_cf, self.init_bw, self.trainable, self.strf_type, self.train_window, self.single_cf
                   )

        return report

