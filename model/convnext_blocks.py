import torch
import torch.nn as nn
from torch import Tensor
from .model_utils import LayerNormChannelLast
from timm.models.layers import DropPath

class ConvNextBlock(nn.Module):
    """
    Standard ConvNeXt encoder block taken from 
    "Liu, Zhuang, et al. "A convnet for the 2020s." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022."
    This implementation is based on Layer normalization. Refer to Fig. 2 of our paper for the definitive structure of this block

    Keyword arguments:
    argument -- 
    dim - dimensions of the input tensor
    drop_path - drop value of the DropPath() function. If its <=0 then an identity function is chosen instead
    layer_scale_init_values - scaling values 

    Return: 
     - None
    """
    def __init__(self, dim: int , drop_path:float = 0., layer_scale_init_value: float=1e-6 ) -> None:
        super(ConvNextBlock, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size =7, padding = 3, groups = dim) #Depth-wise Convolution
        self.norm = LayerNormChannelLast(dim, eps = 1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) #Pointwise or 1x1 Conv layer
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
                                    layer_scale_init_value * torch.ones((dim)),
                                    requires_grad = True) if layer_scale_init_value> 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        forward function 

        Keyword arguments:
        argument -- x : Input torch.Tensor
        Return: x : Output torch.Tensor of the encoder block
        """
        
        input_x = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # Permuting Dimensions --- (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x) #Layer Norm with channel dimensions are aligned at the last 
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # Permuting Dimensions --- (B, H, W, C) -> (B, C, H, W)

        x = input_x + self.drop_path(x) # Add the residual connection
        return x
    
class ConvNextGBlock(nn.Module):
  def __init__(self, input_dim, output_dim, stride, padding):
    super(ConvNextGBlock, self).__init__()

    self.conv_block = nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size = 7, stride = 1, padding = 1, groups = output_dim),
        nn.BatchNorm2d(output_dim),
        nn.Conv2d(output_dim, output_dim, kernel_size = 1, stride = 1, padding = 1),
        nn.GELU(approximate = 'none'),
        nn.Conv2d(output_dim, output_dim, kernel_size = 1, stride = 1, padding = 1),
    )

    self.conv_skip = nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(output_dim),
    )

    self.gelu = nn.GELU(approximate = 'none')

  def forward(self, x):
    return self.gelu(self.conv_block(x) + self.conv_skip(x))