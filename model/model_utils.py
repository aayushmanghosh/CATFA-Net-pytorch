import torch
from torch import nn
from einops import rearrange
from typing import Iterable

#Define LayerNorm2d for building overlappatchmerging module of segformer blocks
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x

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
  


def chunks(data: Iterable, sizes: List[int]):

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