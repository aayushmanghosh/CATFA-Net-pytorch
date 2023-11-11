import torch
from model.catfa_net import CATFANet_Small


def test():
  x = torch.randn((3,3,224,224))
  model = CATFANet_Small(pretrained_encoder_backbone=True)
  preds = model(x)
  print(preds.size())

test()
