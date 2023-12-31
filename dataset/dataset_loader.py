import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from imutils import paths




class SegmentationDatasetWithSplit(Dataset):
  def __init__(self,image_dir, mask_dir, transform = None):
    self.image_dir = image_dir
    self.mask_dir = mask_dir
    self.transform = transform
    self.images = os.listdir(image_dir)

  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
    img_path = os.path.join(self.image_dir, self.images[index])
    mask_path = os.path.join(self.mask_dir, self.images[index])
    image = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"), dtype = np.float32)
    mask[mask == 255.0] = 1.0

    if self.transform is not None:
      augmentations = self.transform(image = image, mask = mask)
      image = augmentations["image"]
      mask = augmentations["mask"]


    return image, mask

class SegmentationDataset(Dataset):
  def __init__(self, imagePaths, maskPaths, transform = None):
    self.transform = transform
    self.imagePaths = imagePaths
    self.maskPaths = maskPaths

  def __len__(self):
    return len(self.imagePaths)

  def __getitem__(self, idx):
    img_path = self.imagePaths[idx]
    mask_path =self.maskPaths[idx]
    image = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"), dtype = np.float32)
    mask[mask == 255.0] = 1.0

    if self.transform is not None:
      augmentations = self.transform(image = image, mask = mask)
      image = augmentations["image"]
      mask = augmentations["mask"]


    return image, mask



def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers = 4,
    pin_memory = True,
):

  train_ds = SegmentationDatasetWithSplit(
      image_dir = train_dir,
      mask_dir = train_maskdir,
      transform = train_transform,
  )

  train_loader = DataLoader(
      train_ds,
      batch_size = batch_size,
      num_workers = num_workers,
      pin_memory = pin_memory,
      shuffle = True,
  )

  val_ds = SegmentationDatasetWithSplit(
      image_dir = val_dir,
      mask_dir = val_maskdir,
      transform = val_transform,
  )

  val_loader = DataLoader(
      val_ds,
      batch_size = batch_size,
      num_workers = num_workers,
      pin_memory = pin_memory,
      shuffle = False,
  )


  return train_loader, val_loader

def get_loaders_with_split(
    trainImages,
    trainMasks,
    testImages,
    testMasks,
    batch_size,
    train_transform,
    val_transform,
    num_workers = 4,
    pin_memory = True,
):

  # create the train and test datasets
  train_ds = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
	  transform=train_transform)
  val_ds = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
    transform=val_transform)

  train_loader = DataLoader(train_ds, shuffle=True,
	  batch_size=batch_size, pin_memory=pin_memory,
	  num_workers = os.cpu_count())
  val_loader = DataLoader(val_ds, shuffle=False,
	  batch_size = batch_size, pin_memory = pin_memory,
	  num_workers = os.cpu_count())

  return train_loader, val_loader