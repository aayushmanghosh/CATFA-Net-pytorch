import torch.nn as nn
import torch
from einops import rearrange
from .model_utils import LayerNorm2d

class ContextAdditionAttention(nn.Module):
  def __init__(self, channels:int, reduction_ratio:int =1, num_heads: int = 8):
    super().__init__()
    self.reducer = nn.Sequential(
        nn.Conv2d(
            channels, channels, kernel_size = reduction_ratio, stride = reduction_ratio
        ),
        LayerNorm2d(channels),
    )

    self.W_qk1 = nn.Conv2d( 2*channels, channels//2, kernel_size = 1, stride = 1, padding = 0)
    self.W_qk2 = nn.Conv2d( channels//2, channels, kernel_size = 1, stride = 1, padding = 0)
    self.Gelu_qk = nn.GELU(approximate = 'none')
    self.att = nn.MultiheadAttention(
        channels, num_heads = num_heads, batch_first=True
    )

  def forward(self, x):
    _,_,h,w = x.shape
    qk = torch.cat((x,x), dim = 1)
    k = self.W_qk2(self.Gelu_qk(self.W_qk1(qk)))
    reduced_k = self.reducer(k)
    reduced_x = self.reducer(x)
    #Attention needs tensor of shape (batch, sequence_length, channels)
    channel_shuffle = "b c h w -> b (h w) c"
    reduced_x = rearrange(reduced_x, channel_shuffle)
    reduced_k = rearrange(reduced_k, channel_shuffle)
    x = rearrange(x, channel_shuffle)
    out = self.att(x, reduced_k, reduced_x)[0]
    #Reshape it back to (batch, channels, height, width)
    channel_original = "b (h w) c -> b c h w"
    out = rearrange(out, channel_original, h=h, w=w)
    return out
  

class CrossChannelTransConvolutionFusionAttention(nn.Module):
  def __init__(self, channels):
    super(CrossChannelTransConvolutionFusionAttention, self).__init__()

    self.W_k = nn.Conv2d(channels, channels, kernel_size = 1, stride =1, padding = 0, bias = True)
    self.W_q = nn.Conv2d(channels, channels, kernel_size = 1, stride = 1, padding = 0, bias = True)
    self.adaptivepool = nn.AdaptiveAvgPool2d((1,1))

    self.W_s = nn.Conv2d(channels, channels, kernel_size = 1, stride = 1, padding = 0)
    self.W_s_maxpool = nn.MaxPool3d((channels,1,1),stride=(channels,1,1)) #Channelwise maxpool
    self.W_s_avgpool = nn.AvgPool3d((channels,1,1), stride = (channels,1,1)) #Channelwise avgpool
    self.W_s_bottle = nn.Sequential(
        nn.Conv2d(2,2,kernel_size = 1, stride = 1, padding = 0),
        nn.GELU(approximate='none'),
        nn.Conv2d(2,2,kernel_size = 3, stride = 1, padding = 1, groups = 2),
        nn.Conv2d(2,1,kernel_size = 1, stride = 1 , padding = 0),
        nn.Sigmoid()
    )

  def crossfeaturechannelattention(self, Q, K, V):
    Q = self.W_q(Q)
    K = self.W_k(K)
    attn_scores = torch.matmul(Q, K.transpose(-2, -1))
    attn_probs = torch.softmax(attn_scores, dim=-1)

    V = self.adaptivepool(V)
    #print(Q.size(), K.size(), V.size())
    out = torch.mul(attn_probs,V)

    return out

  def spatialattention(self,x):
    x1 = self.W_s(x)
    max_x = self.W_s_maxpool(x)
    avg_x = self.W_s_avgpool(x)
    x2 = torch.cat((max_x, avg_x), dim = 1)
    x2 = self.W_s_bottle(x2)
    #print(x1.size(), x2.size())
    out = torch.mul(x1,x2)
    return out

  def forward(self, trans, conv):
    channel_attention = self.crossfeaturechannelattention(trans,conv,trans)
    spatial_attention = self.spatialattention(conv)
    #print(channel_attention.size(), spatial_attention.size())
    return channel_attention + spatial_attention
  
class SpatialFusionAttentionGate(nn.Module):
  def __init__(self,F_g,F_l,F_int):
        super(SpatialFusionAttentionGate,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.gelu = nn.GELU(approximate="none")
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size = 3, stride = 1, padding=1, groups = F_int),
            nn.GELU(approximate = "none"),
            nn.Conv2d(F_int,F_int,kernel_size = 1, stride = 1, padding =0),
            nn.GELU(approximate = "none")
        )
  def forward(self,g,x):
        g1 = self.W_g(g)
        g2 = self.spatial_conv(g1)

        x1 = self.W_x(x)
        psi = self.gelu(g1+x1)
        psi = self.psi(psi)
        x = x1*psi

        return g2+x