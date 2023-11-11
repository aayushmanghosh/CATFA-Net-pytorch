import torch
from torch import nn
from .model_utils import convrelu
from .attention_blocks import CrossChannelTransConvolutionFusionAttention, SpatialFusionAttentionGate
from .context_addition_transformer_encoder import ContextAdditionTransformerEncoder
from .convnext_blocks import ConvNextGBlock
from .convnext_encoder import convnext_build

class CATFANet_Small(nn.Module):
  def __init__(
      self, pretrained_encoder_backbone, n_class = 1, **kwargs
  ):
    super(CATFANet_Small,self).__init__()
    
    self.encoder_trans =  ContextAdditionTransformerEncoder(in_channels=3,
        widths=[96, 192, 384, 768],
        depths=[1,1,3,1],
        all_num_heads=[1, 2, 4, 8],
        patch_sizes=[7, 3, 3, 3],
        overlap_sizes=[4, 2, 2, 2],
        reduction_ratios=[8, 4, 2, 1],
        mlp_expansions=[4, 4, 4, 4],
    )

    self.encoder_conv = convnext_build(
      model_size = 'convnext_tiny',
      pretrained = pretrained_encoder_backbone,
      **kwargs
    )

    self.downsampling_layers = list(self.encoder_conv.downsample_layers.children())
    self.base_layers = list(self.encoder_conv.stages.children())
    self.base_trans= list(self.encoder_trans.stages.children())

    self.layer0 = nn.Sequential(self.downsampling_layers[0], self.base_layers[0])
    self.layer0_trans = nn.Sequential(self.base_trans[0])
    self.layer0_fuse = CrossChannelTransConvolutionFusionAttention(96)
    self.layer0_1x1 = convrelu(96,96,1,0)

    self.layer1 = nn.Sequential(self.downsampling_layers[1], self.base_layers[1])
    self.layer1_trans = nn.Sequential(self.base_trans[1])
    self.layer1_fuse = CrossChannelTransConvolutionFusionAttention(192)
    self.layer1_1x1 = convrelu(192,192,1,0)


    self.layer2 = nn.Sequential(self.downsampling_layers[2], self.base_layers[2])
    self.layer2_trans = nn.Sequential(self.base_trans[2])
    self.layer2_fuse = CrossChannelTransConvolutionFusionAttention(384)
    self.layer2_1x1 = convrelu(384,384,1,0)

    #bottleneck
    self.bottleneck = nn.Sequential(self.downsampling_layers[3])
    self.bott_trans = nn.Sequential(self.base_trans[3])
    self.bott_fuse = CrossChannelTransConvolutionFusionAttention(768)
    self.bott_1x1 = convrelu(768,768,1,0)
    
    self.ups = nn.ModuleList()

    for feature in [384,192,96]:
      self.ups.append(
          nn.ConvTranspose2d(
              feature*2, feature, kernel_size = 2, stride = 2,
          )
      )
      self.ups.append(SpatialFusionAttentionGate(F_g = feature, F_l = feature, F_int = feature))
      self.ups.append(ConvNextGBlock(feature*2, feature,1,1))

    
    self.conv_last = nn.Conv2d(96, n_class, kernel_size = 1)

  def forward(self, input):

    layer0 = self.layer0(input)
    layer1 = self.layer1(layer0)
    layer2 = self.layer2(layer1)

    layer0_conv = self.layer0(input)
    layer0_trans = self.layer0_trans(input)
    layer0 = self.layer0_fuse(trans = layer0_trans, conv= layer0_conv)

    layer1_conv = self.layer1(layer0_conv)
    layer1_trans = self.layer1_trans(layer0)
    layer1 = self.layer1_fuse(trans = layer1_trans, conv= layer1_conv)

    layer2_conv = self.layer2(layer1_conv)
    layer2_trans = self.layer2_trans(layer1)
    layer2 = self.layer2_fuse(trans = layer2_trans, conv= layer2_conv)

    bottleneck = self.bott_trans(layer2)
    bottleneck_conv = self.bottleneck(layer2_conv)
    bottleneck = self.bott_fuse(trans = bottleneck, conv = bottleneck_conv)


    x = self.ups[0](bottleneck)
    layer2 = self.layer2_1x1(layer2)
    layer2 = self.ups[1](g = x, x = layer2)
    x = torch.cat([x,layer2], dim = 1)
    x = self.ups[2](x)

    x = self.ups[3](x)
    layer1 = self.layer1_1x1(layer1)
    layer1 = self.ups[4](g = x, x = layer1)
    x = torch.cat([x,layer1] , dim = 1)
    x = self.ups[5](x)

    x = self.ups[6](x)
    layer0 = self.layer0_1x1(layer0)
    layer0 = self.ups[7](g=x, x=layer0)
    x = torch.cat([x,layer0],dim = 1)
    x = self.ups[8](x)

    mask = self.conv_last(x)
    

    return nn.functional.interpolate(mask, size=(224,224), mode="bilinear", align_corners=False)

class CATFANet_Large(nn.Module):
  def __init__(
      self, pretrained_encoder_backbone, n_class = 1, **kwargs
  ):
    super(CATFANet_Large,self).__init__()
    
    self.encoder_trans =  ContextAdditionTransformerEncoder(in_channels=3,
    widths=[128, 256, 512, 1024],
    depths=[2,2,6,2],
    all_num_heads=[1, 2, 4, 8],
    patch_sizes=[7, 3, 3, 3],
    overlap_sizes=[4, 2, 2, 2],
    reduction_ratios=[8, 4, 2, 1],
    mlp_expansions=[4, 4, 4, 4],
    )

    self.encoder_conv = convnext_build(
      model_size = 'convnext_base',
      pretrained = pretrained_encoder_backbone,
      **kwargs
    )

    self.downsampling_layers = list(self.encoder_conv.downsample_layers.children())
    self.base_layers = list(self.encoder_conv.stages.children())
    self.base_trans= list(self.encoder_trans.stages.children())

    self.layer0 = nn.Sequential(self.downsampling_layers[0], self.encoder_layers[0])
    self.layer0_trans = nn.Sequential(self.base_trans[0])
    self.layer0_fuse = CrossChannelTransConvolutionFusionAttention(128)
    self.layer0_1x1 = convrelu(128,128,1,0)

    self.layer1 = nn.Sequential(self.downsampling_layers[1], self.base_layers[1])
    self.layer1_trans = nn.Sequential(self.base_trans[1])
    self.layer1_fuse = CrossChannelTransConvolutionFusionAttention(256)
    self.layer1_1x1 = convrelu(256,256,1,0)

    self.layer2 = nn.Sequential(self.downsampling_layers[2], self.base_layers[2])
    self.layer2_trans = nn.Sequential(self.base_trans[2])
    self.layer2_fuse = CrossChannelTransConvolutionFusionAttention(512)
    self.layer2_1x1 = convrelu(512,512,1,0)

    #bottleneck
    self.bottleneck = nn.Sequential(self.downsampling_layers[3])
    self.bott_trans = nn.Sequential(self.base_trans[3])
    self.bott_fuse = CrossChannelTransConvolutionFusionAttention(1024)
    self.bott_1x1 = convrelu(1024,1024,1,0)
    
    self.ups = nn.ModuleList()

    for feature in [512,256,128]:
      self.ups.append(
          nn.ConvTranspose2d(
              feature*2, feature, kernel_size = 2, stride = 2,
          )
      )
      self.ups.append(SpatialFusionAttentionGate(F_g = feature, F_l = feature, F_int = feature))
      self.ups.append(ConvNextGBlock(feature*2, feature,1,1))

    
    self.conv_last = nn.Conv2d(128, n_class, kernel_size = 1)

  def forward(self, input):

    layer0 = self.layer0(input)
    layer1 = self.layer1(layer0)
    layer2 = self.layer2(layer1)

    layer0_conv = self.layer0(input)
    layer0_trans = self.layer0_trans(input)
    layer0 = self.layer0_fuse(trans = layer0_trans, conv= layer0_conv)

    layer1_conv = self.layer1(layer0_conv)
    layer1_trans = self.layer1_trans(layer0)
    layer1 = self.layer1_fuse(trans = layer1_trans, conv= layer1_conv)

    layer2_conv = self.layer2(layer1_conv)
    layer2_trans = self.layer2_trans(layer1)
    layer2 = self.layer2_fuse(trans = layer2_trans, conv= layer2_conv)

    bottleneck = self.bott_trans(layer2)
    bottleneck_conv = self.bottleneck(layer2_conv)
    bottleneck = self.bott_fuse(trans = bottleneck, conv = bottleneck_conv)


    x = self.ups[0](bottleneck)
    layer2 = self.layer2_1x1(layer2)
    layer2 = self.ups[1](g = x, x = layer2)
    x = torch.cat([x,layer2], dim = 1)
    x = self.ups[2](x)

    x = self.ups[3](x) #upsample1
    layer1 = self.layer1_1x1(layer1)
    layer1 = self.ups[4](g = x, x = layer1)
    x = torch.cat([x,layer1] , dim = 1)
    x = self.ups[5](x)

    x = self.ups[6](x)
    layer0 = self.layer0_1x1(layer0)
    layer0 = self.ups[7](g=x, x=layer0)
    x = torch.cat([x,layer0],dim = 1)
    x = self.ups[8](x)

    mask = self.conv_last(x)
    

    return nn.functional.interpolate(mask, size=(224,224), mode="bilinear", align_corners=False)