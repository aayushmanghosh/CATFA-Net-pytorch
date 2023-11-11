import torch
from torch import nn
from typing import List
from .model_utils import LayerNorm2d, PatchMerging, chunks
from .context_addition_transformer_block import ContextAdditionTransformerBlock


class ContextAdditionTransformerStage(nn.Sequential):
  def __init__(
      self,
      in_channels:int,
      out_channels: int,
      patch_size: int,
      overlap_size: int,
      drop_probs: List[int],
      depth: int = 2,
      reduction_ratio: int = 1,
      num_heads: int = 8,
      mlp_expansion: int = 4,
  ):

    super().__init__()
    self.overlap_patch_merge = PatchMerging(
        in_channels, out_channels, patch_size, overlap_size,
    )
    self.blocks = nn.Sequential(
        *[
            ContextAdditionTransformerBlock(
                out_channels, reduction_ratio, num_heads, mlp_expansion, drop_probs[i]
            )
            for i in range(depth)

        ]
    )

    self.norm = LayerNorm2d(out_channels)


class ContextAdditionTransformerEncoder(nn.Module):
  def __init__(
        self,
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansions: List[int],
        drop_prob: float = .0
    ):
        super().__init__()
        # create drop paths probabilities (one for each stage's block)
        drop_probs =  [x.item() for x in torch.linspace(0, drop_prob, sum(depths))]
        self.stages = nn.ModuleList(
            [
                ContextAdditionTransformerStage(*args)
                for args in zip(
                    [in_channels, *widths],
                    widths,
                    patch_sizes,
                    overlap_sizes,
                    chunks(drop_probs, sizes=depths),
                    depths,
                    reduction_ratios,
                    all_num_heads,
                    mlp_expansions
                )
            ]
        )

  def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features