"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F
import math
import matplotlib.pyplot as plt

from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import eval_np
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm


def identity(x):
    return x


class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size
        self.img_encoder = ShallowConv()
        
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        # print("in Forward!!!")

        # print("input.shape: ", input.shape)
        batch_size = input.shape[0]
        # plt.imshow(input[:,:-39].reshape(batch_size,256,256,3)[2].type(torch.int64))
        # plt.show()
        # 39 because this is critic and takes obs+action as input, i.e. :32 for the image obs + :7 for the action
        img_feat = self.img_encoder(input[:,:-49].reshape(batch_size,256,256,3).permute(0,3,1,2))
        h = torch.cat((img_feat, input[:, -49:]), axis=1)
        
        # h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return eval_np(self, obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


class ConvBase(torch.nn.Module):
    """
    Base class for ConvNets.
    """
    def __init__(self):
        super(ConvBase, self).__init__()

    # dirty hack - re-implement to pass the buck onto subclasses from ABC parent
    # def output_shape(self, input_shape):
    #     """
    #     Function to compute output shape from inputs to this module. 

    #     Args:
    #         input_shape (iterable of int): shape of input. Does not include batch dimension.
    #             Some modules may not need this argument, if their output does not depend 
    #             on the size of the input, or if they assume fixed size input.

    #     Returns:
    #         out_shape ([int]): list of integers corresponding to output shape
    #     """
    #     raise NotImplementedError

    def forward(self, inputs):
        x = self.nets(inputs)
        # if list(self.output_shape(list(inputs.shape)[1:])) != list(x.shape)[1:]:
        #     raise ValueError('Size mismatch: expect size %s, but got size %s' % (
        #         str(self.output_shape(list(inputs.shape)[1:])), str(list(x.shape)[1:]))
        #     )
        return x


class ShallowConv(ConvBase):
    """
    A shallow convolutional encoder from https://rll.berkeley.edu/dsae/dsae.pdf
    """
    def __init__(self, input_channel=3, output_channel=1024):
        super(ShallowConv, self).__init__()
        self._input_channel = input_channel
        self._output_channel = output_channel
        self.nets = nn.Sequential(
            torch.nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(1024)
        )

    # def output_shape(self, input_shape):
    #     """
    #     Function to compute output shape from inputs to this module. 

    #     Args:
    #         input_shape (iterable of int): shape of input. Does not include batch dimension.
    #             Some modules may not need this argument, if their output does not depend 
    #             on the size of the input, or if they assume fixed size input.

    #     Returns:
    #         out_shape ([int]): list of integers corresponding to output shape
    #     """
    #     assert(len(input_shape) == 3)
    #     assert(input_shape[0] == self._input_channel)
    #     out_h = int(math.floor(input_shape[1] / 2.))
    #     out_w = int(math.floor(input_shape[2] / 2.))
    #     return [self._output_channel, out_h, out_w]