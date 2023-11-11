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

#Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
BATCH_SIZE = 8
NUM_EPOCHS = 20
NUM_WORKERS = 2
IMAGE_HEIGHT = 224
IMAGE_WIDTH  = 224
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/mnt/d/refuge_cropped_final/refuge_cropped_disc/train/"
TRAIN_MASK_DIR = "/mnt/d/refuge_cropped_final/refuge_cropped_disc/train_masks/"
VAL_IMG_DIR = "/mnt/d/refuge_cropped_final/refuge_cropped_disc/test/"
VAL_MASK_DIR = "/mnt/d/refuge_cropped_final/refuge_cropped_disc/test_masks/"

# base path of the dataset
DATASET_PATH = "/mnt/d/medical_image_datasets/CVCClinicDB/PNG_resized/"
# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")
# define the test split
TEST_SPLIT = 0.20


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
  

  
  model = CATFANet_Small(pretrained_encoder_backbone=True).to(DEVICE)
  #model = CATFANet_Large(convnext_base(weights = 'IMAGENET1K_V1')).to(DEVICE)
  #model = RFAUConvNext_Tiny(convnext_tiny(weights = 'IMAGENET1K_V1')).to(DEVICE)
  loss_fn = DiceLoss() #For multiclass change to crossentropy
  optimizer = optim.AdamW(model.parameters(), lr= LEARNING_RATE)

  train_loader, val_loader = get_loaders(
      TRAIN_IMG_DIR,
      TRAIN_MASK_DIR,
      VAL_IMG_DIR,
      VAL_MASK_DIR,
      BATCH_SIZE,
      train_transform,
      val_transforms
  )
  

#   train_loader, val_loader = get_loaders_with_split(
#       image_dataset_path= IMAGE_DATASET_PATH,
#       mask_dataset_path= MASK_DATASET_PATH,
#       batch_size=BATCH_SIZE,
#       test_split= TEST_SPLIT,
#       train_transform=train_transform,
#       val_transform=val_transforms
#   )

  scaler = torch.cuda.amp.GradScaler()
  print(len(val_loader))

  for epoch in range(NUM_EPOCHS):
    print("Epoch:",epoch)
    train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)

    #save model
    # checkpoint = {
    #     "state_dict":model.state_dict(),
    #     "optimizer":optimizer.state_dict(),
    # }
    # save_checkpoint(checkpoint)
    #check accuracy
    test_one_epoch(val_loader, model, train_loss, loss_fn, device =DEVICE)

    #print some examples
    # save_predictions_as_imgs(
    #     val_loader, model, folder = "", device =DEVICE
    # )

  image_paths = sorted(list(paths.list_images(IMAGE_DATASET_PATH)))
  mask_paths = sorted(list(paths.list_images(MASK_DATASET_PATH)))
  # partition the data into training and testing splits using 85% of
  # the data for training and the remaining 15% for testing
  split = train_test_split(image_paths, mask_paths,
    test_size=TEST_SPLIT, random_state=42)


  # unpack the data split
  (test_images, test_masks) = split[2:]

  #Save data for ROC curve
  val_ds = SegmentationDataset(imagePaths=test_images, maskPaths=test_masks,
    transform=val_transforms)
  

  val_loader_new = DataLoader(val_ds, shuffle=False,
	  batch_size = len(test_images), pin_memory = True,
	  num_workers = os.cpu_count())
#   fpr, tpr, _ = compute_roc(val_loader_new, model)
#   roc_curve = {'FPR': fpr.tolist(),
#                'TPR': tpr.tolist()}
  
#   df_roc = pd.DataFrame(roc_curve)
#   df_roc.to_csv('/content/gdrive/MyDrive/final_results_transconv/chase_transconv_small/roc_curve.csv')

#   #Save data for PR curve
#   prec, rec, _ = compute_prcurve(val_loader_new, model)
#   pr_curve = {'Precision': prec.tolist(),
#                'Recall': rec.tolist()}
               
#   df_pr = pd.DataFrame(pr_curve)
#   df_pr.to_csv('/content/gdrive/MyDrive/final_results_transconv/chase_transconv_small/pr_curve.csv')

if __name__ == '__main__':
    train_test_setup()
    metric = get_metrics_data()
    df = pd.DataFrame(metric)
    df.to_csv(os.path.join('metric_results.csv'))