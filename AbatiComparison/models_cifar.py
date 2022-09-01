import torch
import torchvision
import torch.nn as nn

import numpy as np
from functools import reduce
from operator import mul
from typing import Tuple

def residual_op(x, functions, bns, activation_fn):
    # type: (torch.Tensor, List[Module, Module, Module], List[Module, Module, Module], Module) -> torch.Tensor
    """
    Implements a global residual operation.
    :param x: the input tensor.
    :param functions: a list of functions (nn.Modules).
    :param bns: a list of optional batch-norm layers.
    :param activation_fn: the activation to be applied.
    :return: the output of the residual operation.
    """
    f1, f2, f3 = functions
    bn1, bn2, bn3 = bns

    assert len(functions) == len(bns) == 3
    assert f1 is not None and f2 is not None
    assert not (f3 is None and bn3 is not None)

    # A-branch
    ha = x
    ha = f1(ha)
    if bn1 is not None:
        ha = bn1(ha)
    ha = activation_fn(ha)

    ha = f2(ha)
    if bn2 is not None:
        ha = bn2(ha)

    # B-branch
    hb = x
    if f3 is not None:
        hb = f3(hb)
    if bn3 is not None:
        hb = bn3(hb)

    # Residual connection
    out = ha + hb
    return activation_fn(out)

class BaseModule(nn.Module):
    """
    Implements the basic module.
    All other modules inherit from this one
    """
    def load_w(self, checkpoint_path):
        # type: (str) -> None
        """
        Loads a checkpoint into the state_dict.
        :param checkpoint_path: the checkpoint file to be loaded.
        """
        self.load_state_dict(torch.load(checkpoint_path))

    def __repr__(self):
        # type: () -> str
        """
        String representation
        """
        good_old = super(BaseModule, self).__repr__()
        addition = 'Total number of parameters: {:,}'.format(self.n_parameters)

        return good_old + '\n' + addition

    def __call__(self, *args, **kwargs):
        return super(BaseModule, self).__call__(*args, **kwargs)

    @property
    def n_parameters(self):
        # type: () -> int
        """
        Number of parameters of the model.
        """
        n_parameters = 0
        for p in self.parameters():
            if hasattr(p, 'mask'):
                n_parameters += torch.sum(p.mask).item()
            else:
                n_parameters += reduce(mul, p.shape)
        return int(n_parameters)


class BaseBlock(BaseModule):
    """ Base class for all blocks. """
    def __init__(self, channel_in, channel_out, activation_fn, use_bn=True, use_bias=False):
        # type: (int, int, Module, bool, bool) -> None
        """
        Class constructor.
        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(BaseBlock, self).__init__()

        assert not (use_bn and use_bias), 'Using bias=True with batch_normalization is forbidden.'

        self._channel_in = channel_in
        self._channel_out = channel_out
        self._activation_fn = activation_fn
        self._use_bn = use_bn
        self._bias = use_bias

    def get_bn(self):
        # type: () -> Optional[Module]
        """
        Returns batch norm layers, if needed.
        :return: batch norm layers or None
        """
        return nn.BatchNorm2d(num_features=self._channel_out) if self._use_bn else None

    def forward(self, x):
        """
        Abstract forward function. Not implemented.
        """
        raise NotImplementedError


class DownsampleBlock(BaseBlock):
    """ Implements a Downsampling block for images (Fig. 1ii). """
    def __init__(self, channel_in, channel_out, activation_fn, use_bn=True, use_bias=False):
        # type: (int, int, Module, bool, bool) -> None
        """
        Class constructor.
        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(DownsampleBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)

        # Convolutions
        self.conv1a = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3,
                                padding=1, stride=2, bias=use_bias)
        self.conv1b = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                                padding=1, stride=1, bias=use_bias)
        self.conv2a = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=1,
                                padding=0, stride=2, bias=use_bias)

        # Batch Normalization layers
        self.bn1a = self.get_bn()
        self.bn1b = self.get_bn()
        self.bn2a = self.get_bn()

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the input tensor
        :return: the output tensor
        """
        return residual_op(
            x,
            functions=[self.conv1a, self.conv1b, self.conv2a],
            bns=[self.bn1a, self.bn1b, self.bn2a],
            activation_fn=self._activation_fn
        )


class UpsampleBlock(BaseBlock):
    """ Implements a Upsampling block for images (Fig. 1ii). """
    def __init__(self, channel_in, channel_out, activation_fn, use_bn=True, use_bias=False):
        # type: (int, int, Module, bool, bool) -> None
        """
        Class constructor.
        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(UpsampleBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)

        # Convolutions
        self.conv1a = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5,
                                         padding=2, stride=2, output_padding=1, bias=use_bias)
        self.conv1b = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                                padding=1, stride=1, bias=use_bias)
        self.conv2a = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=1,
                                         padding=0, stride=2, output_padding=1, bias=use_bias)

        # Batch Normalization layers
        self.bn1a = self.get_bn()
        self.bn1b = self.get_bn()
        self.bn2a = self.get_bn()

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the input tensor
        :return: the output tensor
        """
        return residual_op(
            x,
            functions=[self.conv1a, self.conv1b, self.conv2a],
            bns=[self.bn1a, self.bn1b, self.bn2a],
            activation_fn=self._activation_fn
        )


class ResidualBlock(BaseBlock):
    """ Implements a Residual block for images (Fig. 1ii). """
    def __init__(self, channel_in, channel_out, activation_fn, use_bn=True, use_bias=False):
        # type: (int, int, Module, bool, bool) -> None
        """
        Class constructor.
        :param channel_in: number of input channels.
        :param channel_out: number of output channels.
        :param activation_fn: activation to be employed.
        :param use_bn: whether or not to use batch-norm.
        :param use_bias: whether or not to use bias.
        """
        super(ResidualBlock, self).__init__(channel_in, channel_out, activation_fn, use_bn, use_bias)

        # Convolutions
        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3,
                               padding=1, stride=1, bias=use_bias)
        self.conv2 = nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=3,
                               padding=1, stride=1, bias=use_bias)

        # Batch Normalization layers
        self.bn1 = self.get_bn()
        self.bn2 = self.get_bn()

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the input tensor
        :return: the output tensor
        """
        return residual_op(
            x,
            functions=[self.conv1, self.conv2, None],
            bns=[self.bn1, self.bn2, None],
            activation_fn=self._activation_fn
        )

class Encoder(BaseModule):
    """
    CIFAR10 model encoder.
    """
    def __init__(self, input_shape, code_length):
        # type: (Tuple[int, int, int], int) -> None
        """
        Class constructor:
        :param input_shape: the shape of CIFAR10 samples.
        :param code_length: the dimensionality of latent vectors.
        """
        super(Encoder, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length

        c, h, w = input_shape

        activation_fn = nn.LeakyReLU()

        # Convolutional network
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, bias=False),
            activation_fn,
            ResidualBlock(channel_in=32, channel_out=32, activation_fn=activation_fn),
            DownsampleBlock(channel_in=32, channel_out=64, activation_fn=activation_fn),
            DownsampleBlock(channel_in=64, channel_out=128, activation_fn=activation_fn),
            DownsampleBlock(channel_in=128, channel_out=256, activation_fn=activation_fn),
        )
        self.deepest_shape = (256, h // 8, w // 8)

        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=reduce(mul, self.deepest_shape), out_features=256),
            nn.BatchNorm1d(num_features=256),
            activation_fn,
            nn.Linear(in_features=256, out_features=code_length),
            nn.Sigmoid()
        )

    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the input batch of images.
        :return: the batch of latent vectors.
        """
        h = x
        h = self.conv(h)
        h = h.view(len(h), -1)
        o = self.fc(h)

        return o


class Decoder(BaseModule):
    """
    CIFAR10 model decoder.
    """
    def __init__(self, code_length, deepest_shape, output_shape):
        # type: (int, Tuple[int, int, int], Tuple[int, int, int]) -> None
        """
        Class constructor.
        :param code_length: the dimensionality of latent vectors.
        :param deepest_shape: the dimensionality of the encoder's deepest convolutional map.
        :param output_shape: the shape of CIFAR10 samples.
        """
        super(Decoder, self).__init__()

        self.code_length = code_length
        self.deepest_shape = deepest_shape
        self.output_shape = output_shape

        activation_fn = nn.LeakyReLU()

        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=code_length, out_features=256),
            nn.BatchNorm1d(num_features=256),
            activation_fn,
            nn.Linear(in_features=256, out_features=reduce(mul, deepest_shape)),
            nn.BatchNorm1d(num_features=reduce(mul, deepest_shape)),
            activation_fn
        )

        # Convolutional network
        self.conv = nn.Sequential(
            UpsampleBlock(channel_in=256, channel_out=128, activation_fn=activation_fn),
            UpsampleBlock(channel_in=128, channel_out=64, activation_fn=activation_fn),
            UpsampleBlock(channel_in=64, channel_out=32, activation_fn=activation_fn),
            ResidualBlock(channel_in=32, channel_out=32, activation_fn=activation_fn),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the batch of latent vectors.
        :return: the batch of reconstructions.
        """
        h = x
        h = self.fc(h)
        h = h.view(len(h), *self.deepest_shape)
        h = self.conv(h)
        o = h

        return o
    
class Model(nn.Module):
    def __init__(self, code_length):
        super(Model, self).__init__()
        self.encoder = Encoder((3,32,32),code_length=code_length)
        self.decoder = Decoder(code_length=code_length, deepest_shape=self.encoder.deepest_shape, output_shape=(3,32,32))
        
    def forward(self,x):
        return self.decoder(self.encoder(x))
    
    
class ToFloatTensor2D(object):
    """ Convert images to FloatTensors """
    def __init__(self,device):
        self.device=device

    def __call__(self, sample):
        X = sample

        X = np.array(X)
        #Y = np.array(Y)
        #X = np.expand_dims(X,-1)
        # swap color axis because
        # numpy image: B x H x W x C
        X = X/ 255.
        X = X.transpose(2, 0, 1)

        X = np.float32(X)
        #Y = np.float32(Y)
        return torch.from_numpy(X).to(self.device)#, torch.from_numpy(Y)
    
    

