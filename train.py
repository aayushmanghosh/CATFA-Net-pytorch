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
    if args.pretrained == "True":
        model = CATFANet_Small(pretrained_encoder_backbone=True).to(DEVICE)
    else:
        model = CATFANet_Small(pretrained_encoder_backbone=False).to(DEVICE)
    
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