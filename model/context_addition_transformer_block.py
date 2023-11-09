from torch import nn
from model_utils import ResidualAdd, LayerNorm2d
from attention_blocks import ContextAdditionAttention
from torchvision.ops import StochasticDepth

class ConvFFN(nn.Sequential):
  def __init__(self, channels:int, expansion:int = 4):
    super().__init__(
        #dense layer
        nn.Conv2d(channels, channels, kernel_size = 1),
        #depth wise conv
        nn.Conv2d(
            channels,
            channels * expansion,
            kernel_size = 3,
            groups = channels,
            padding  = 1,
        ),
        nn.GELU(),
        #dense layer
        nn.Conv2d(channels*expansion, channels, kernel_size = 1)
    )

class ContextAdditionTransformerBlock(nn.Sequential):
  def __init__(
      self,
      channels:int,
      reduction_ratio:int = 1,
      num_heads: int = 8,
      mlp_expansion: int=4,
      drop_path_prob: float = .0
  ):

    super().__init__(
        ResidualAdd(
            nn.Sequential(
                LayerNorm2d(channels),
                ContextAdditionAttention(channels, reduction_ratio, num_heads),
            )
        ),
        ResidualAdd(
            nn.Sequential(
                LayerNorm2d(channels),
                ConvFFN(channels, expansion = mlp_expansion),
                StochasticDepth(p=drop_path_prob, mode = "batch")
            )
        ),
    )