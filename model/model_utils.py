import torch
from torch import nn, Tensor
from einops import rearrange
from typing import Iterable
import torch.nn.functional as F

#Define LayerNorm2d for building overlappatchmerging module of CAT blocks
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x

class LayerNorm(nn.Module):
    """Implementation of layer normalization for the data format: 
    channels_first. Thus the dimensions of the input and the output will be
    (batch_size, channels, height, width)
    
    Keyword arguments:
    
    arguments -- 
    normalized_dim - dimension of the tensors to be normalized
    eps - epsilon value for Layer Normalization equation

    Return: Normalized value of input tensor x.
    """
    
    def __init__(self, normalized_dim, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_dim))
        self.bias = nn.Parameter(torch.zeros(normalized_dim))
        self.eps = eps
        self.normalized_dim = (normalized_dim, )

    def forward(self, x):
        u = x.mean(1, keepdims = True)
        s = (x-u).pow(2).mean(1, keepdim = True)
        x = (x-u)/ torch.sqrt( s + self.eps)
        x = self.weight[:,None, None]*x + self.bias[:,None, None]
        return x

class LayerNormChannelLast(nn.Module):

    """Implementation of layer normalization for the data format: 
    channels_last. Thus the dimensions of the input and the output will be
    (batch_size, height, width, channels)
    
    Keyword arguments:
    
    arguments -- 
    normalized_dim - dimension of the tensors to be normalized
    eps - epsilon value for Layer Normalization equation

    Return: Normalized value of input tensor x."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super(LayerNormChannelLast, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x: Tensor) -> Tensor:
        #Just call layer norm from nn.Functional
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

# Define OverlapPatchMerging which is just a conv layer followed by LayerNorm2d
# The Patch Merging layer involves different patches sizes throughout the encoder length.
class PatchMerging(nn.Sequential):
  def __init__(
      self, in_channels:int, out_channels:int, patch_size:int, overlap_size:int
  ):

    super().__init__(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size = patch_size,
            stride = overlap_size,
            padding = patch_size//2,
            bias = False
        ),
        LayerNorm2d(out_channels)
    )

class ResidualAdd(nn.Module):
  """Just an util Layer"""
  def __init__(self, fn):
    super().__init__()
    self.fn = fn

  def forward(self, x, **kwargs):
    out = self.fn(x, **kwargs)
    x = x + out
    return x
  


def chunks(data: Iterable, sizes):

  """ Given an iterable, returns slices using sizes as indices """
  curr = 0
  for size in sizes:
    chunk = data[curr:curr+size]
    curr+=size
    yield chunk

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class Stem_Op(nn.Module):
  def __init__(self):
    super(Stem_Op, self).__init__()

    self.stem = nn.Sequential(
        nn.Conv2d(3,64,kernel_size = 4, stride = 4, padding = 1),
        LayerNorm2d(64),
    )

  def forward(self, x):
    return self.stem(x)


class DownSample(nn.Module):
  def __init__(self, inp, out):
    super(DownSample, self).__init__()
    self.down = nn.Sequential(
        LayerNorm2d(inp),
        nn.Conv2d(inp, out, kernel_size = 2, stride = 2),
    )
  def forward(self,x):
    return self.down(x)