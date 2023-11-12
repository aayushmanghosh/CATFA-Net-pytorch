from utils import *
from model.catfa_net import CATFANet_Small, CATFANet_Large
from dataset.dataset_loader import get_loaders, get_loaders_with_split, SegmentationDataset
from sklearn.model_selection import train_test_split
from imutils import paths
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import pandas as pd
import os
import torch.optim as optim
from torch import nn
from torchvision.models import convnext_tiny
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('dir', type = str, help = "directory of dataset")
parser.add_argument('--pre_split', '-ps', type = str, default = "True",
                    help = "Whether dataset has predefined split or not")
parser.add_argument('--split', '-s', type = float, default = 0.20, 
                    help = "Mention test split ratio. Defaults to 0.20")
parser.add_argument('--result_dir', '-r', type = str, default= 'results' , help = "directory for storing results")
parser.add_argument('--save', type = str, default="False", help = "save model option for saving checkpoints at each epoch")
parser.add_argument('--save_file_name', type= str, default='catfanet_model.pth.tar', help = "Model save name. Save format is .pth.tar")

parser.add_argument('--num_epochs', '-e', type = int, 
                    default= 50, help = "number of training epochs")
parser.add_argument('--lr', type = float, 
                    default= 1e-4, help = "learning rate for optimizer")
parser.add_argument('--batch_size', '-B', type = int,
                    default= 16, help = "batch size per training epoch")
parser.add_argument('--model_size', '-m', type = str, 
                    default= 'small', help = "model size of CATFA-Net")
parser.add_argument('--num_workers', '-w', type = int, default=2, help="number of cpu workers for training")
parser.add_argument('--pin_mem', type = str, default="True", help ="pin memory for dataset loaders" )
parser.add_argument('--optimizer', type = str, default = 'adamw', help = "gradient optimizer.")
parser.add_argument('--pretrained', '-p', type = str, default = "True", help = "enable/disable pretrained convnext encoder backbone")
parser.add_argument('--save_roc_pr', type = str, default="False", help = "Whether to save ROC and PR curve information for plotting it in origin")

args = parser.parse_args()



#Hyperparameters etc.
LEARNING_RATE = args.lr
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
NUM_WORKERS = args.num_workers
IMAGE_HEIGHT = 224
IMAGE_WIDTH  = 224
PIN_MEMORY = False
if args.pin_mem == "True":
  PIN_MEMORY = True



class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DoubleConv, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True),
    )
  def forward(self,x):
    return self.conv(x)

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class ConvNextDecBlock(nn.Module):
  def __init__(self, input_dim, output_dim, stride, padding):
    super(ConvNextDecBlock, self).__init__()

    self.conv_block = nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size = 7, stride = 1, padding = 1, groups = output_dim),
        nn.BatchNorm2d(output_dim),
        nn.Conv2d(output_dim, output_dim, kernel_size = 1, stride = 1, padding = 1),
        nn.GELU(approximate = 'none'),
        nn.Conv2d(output_dim, output_dim, kernel_size = 1, stride = 1, padding = 1),
        nn.BatchNorm2d(output_dim),
    )

    self.conv_skip = nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(output_dim),
    )

    self.gelu = nn.GELU(approximate = 'none')

  def forward(self, x):
    return self.gelu(self.conv_block(x) + self.conv_skip(x))
  
class AttentionGate(nn.Module):
  def __init__(self,F_g,F_l,F_int):
        super(AttentionGate,self).__init__()
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

        self.relu = nn.GELU(approximate="none")

  def forward(self,g,x):
        g1 = self.W_g(g)
        g2 = self.relu(g1)

        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        x = x*psi

        return g2+x
  
class RFAUConvNext_Tiny(nn.Module):
  def __init__(
      self, encoder, n_class = 1
  ):
    super(RFAUConvNext_Tiny,self).__init__()
    #Down part of convunext
    self.encoder = encoder
    self.base_layers = list(self.encoder.features.children())
    self.layer0 = nn.Sequential(*self.base_layers[:2])
    self.layer0_1x1 = convrelu(96,96,1,0) #out = 128
    self.layer1 = nn.Sequential(*self.base_layers[2:4])
    self.layer1_1x1 = convrelu(192,192,1,0) #out = 256
    self.layer2 = nn.Sequential(*self.base_layers[4:6])
    self.layer2_1x1 = convrelu(384,384,1,0) #out = 512

    #bottleneck
    self.bottleneck = nn.Sequential(*self.base_layers[6:]) #out = 1024
    self.bott_1x1 = convrelu(768,768,1,0)
    #Up part of convunext
    self.ups = nn.ModuleList()

    for feature in [384,192,96]:
      self.ups.append(
          nn.ConvTranspose2d(
              feature*2, feature, kernel_size = 2, stride = 2,
          )
      )
      self.ups.append(AttentionGate(F_g = feature, F_l = feature, F_int = feature))
      self.ups.append(ConvNextDecBlock(feature*2, feature,1,1))

    #Last conv layer
    self.conv_last = nn.Conv2d(96, n_class, kernel_size = 1)

  def forward(self, input):

    layer0 = self.layer0(input)
    layer1 = self.layer1(layer0)
    layer2 = self.layer2(layer1)

    bottleneck = self.bottleneck(layer2)
    bottleneck = self.bott_1x1(bottleneck)

    x = self.ups[0](bottleneck) #upsample 2
    layer2 = self.layer2_1x1(layer2)
    layer2 = self.ups[1](g = x, x = layer2)
    x = torch.cat([x,layer2], dim = 1)
    x = self.ups[2](x) #Double Convolutions

    x = self.ups[3](x) #upsample1
    layer1 = self.layer1_1x1(layer1)
    layer1 = self.ups[4](g = x, x = layer1)
    x = torch.cat([x,layer1] , dim = 1)
    x = self.ups[5](x) #Double Convolutions

    x = self.ups[6](x)
    layer0 = self.layer0_1x1(layer0)
    layer0 = self.ups[7](g=x, x=layer0)
    x = torch.cat([x,layer0],dim = 1)
    x = self.ups[8](x)

    mask = self.conv_last(x)
    # preds = torch.sigmoid(mask)
    # preds = (preds > 0.5).float()

    # for x in range(preds.size(dim = 0)):
    #   mask[x] = numIslands(mask[x],preds[x])
    # del preds

    return nn.functional.interpolate(mask, size=(224,224), mode="bilinear", align_corners=False)

def split_and_load(test_split):
  image_paths = sorted(list(paths.list_images(args.dir+ '/images')))
  mask_paths = sorted(list(paths.list_images(args.dir+ '/masks')))
  split = train_test_split(image_paths, mask_paths,
    test_size=test_split, random_state=42)


  # unpack the data split
  (train_images, test_images) = split[:2]
  (train_masks, test_masks) = split[2:]

  return train_images, test_images, train_masks, test_masks


def train_test_setup():
  train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
  val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
  
  model = None

  if args.model_size == 'small':
    # if args.pretrained == "True":
    #     model = CATFANet_Small(pretrained_encoder_backbone=True).to(DEVICE)
    # else:
    #     model = CATFANet_Small(pretrained_encoder_backbone=False).to(DEVICE)
    model = RFAUConvNext_Tiny(convnext_tiny(weights = 'IMAGENET1K_V1')).to(DEVICE)
  else:
     if args.pretrained == "True":
        model = CATFANet_Large(pretrained_encoder_backbone=True).to(DEVICE)
     else:
        model = CATFANet_Large(pretrained_encoder_backbone=False).to(DEVICE)
     
  
  
  loss_fn = DiceLoss() #For multiclass change to crossentropy

  optimizer = optim.AdamW(model.parameters(), lr= LEARNING_RATE)
  if args.optimizer == 'sgd' or args.optimizer =='SGD':
     optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE)
  
  else:
     optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

  

  if args.pre_split == "True":
     train_loader, val_loader = get_loaders(
      args.dir + '/train',
      args.dir + '/train_masks',
      args.dir + '/test',
      args.dir + '/test_masks',
      BATCH_SIZE,
      train_transform,
      val_transforms,
      num_workers= NUM_WORKERS,
      pin_memory= PIN_MEMORY
    )
  
  if args.pre_split == "False":
    train_images, test_images, train_masks, test_masks = split_and_load(args.split)

    train_loader, val_loader = get_loaders_with_split(
        train_images,
        train_masks,
        test_images,
        test_masks,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        num_workers= NUM_WORKERS,
        pin_memory= PIN_MEMORY
    )

  scaler = torch.cuda.amp.GradScaler()
  print(len(val_loader))

  for epoch in range(NUM_EPOCHS):
    print("Epoch:",epoch)
    train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)

    if args.save == "True":
        checkpoint = {
          "state_dict":model.state_dict(),
          "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename = args.save_file_name)
    
    #check accuracy
    test_one_epoch(val_loader, model, train_loss, loss_fn, device =DEVICE)

    #print some examples
    save_predictions_as_imgs(
        val_loader, model, folder = args.result_dir, device =DEVICE
    )

  if args.save_roc_pr == "True":
     train_images, test_images, train_masks, test_masks = split_and_load(args.split)
     #Save data for ROC curve
     val_ds = SegmentationDataset(imagePaths = test_images, maskPaths=test_masks,
                  transform = val_transforms)
     val_loader_new = DataLoader(val_ds, shuffle=False,
	      batch_size = len(test_images), pin_memory = PIN_MEMORY,
	      num_workers = NUM_WORKERS)

     fpr, tpr, _ = compute_roc(val_loader_new, model)
     roc_curve = {'FPR': fpr.tolist(),
                 'TPR': tpr.tolist()}
    
     df_roc = pd.DataFrame(roc_curve)
     df_roc.to_csv(args.result_dir + '/roc_curve.csv')

     #Save data for PR curve
     prec, rec, _ = compute_prcurve(val_loader_new, model)
     pr_curve = {'Precision': prec.tolist(),
                 'Recall': rec.tolist()}
                
     df_pr = pd.DataFrame(pr_curve)
     df_pr.to_csv(args.result_dir + '/pr_curve.csv')

if __name__ == '__main__':
    train_test_setup()
    metric = get_metrics_data()
    df = pd.DataFrame(metric)
    df.to_csv(args.result_dir + '/report.csv')