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
    image_dataset_path,
    mask_dataset_path,
    batch_size,
    train_transform,
    val_transform,
    test_split,
    pin_memory = True,
):
  image_paths = sorted(list(paths.list_images(image_dataset_path)))
  mask_paths = sorted(list(paths.list_images(mask_dataset_path)))
  # partition the data into training and testing splits using 85% of
  # the data for training and the remaining 15% for testing
  split = train_test_split(image_paths, mask_paths,
    test_size=test_split, random_state=42)


  # unpack the data split
  (train_images, train_masks) = split[:2]
  (test_images, test_masks) = split[2:]
  print(len(test_images))
  # create the train and test datasets
  train_ds = SegmentationDataset(
    imagePaths=train_images, 
    maskPaths=train_masks,
	transform=train_transform
    )
  
  val_ds = SegmentationDataset(
    imagePaths=test_images, 
    maskPaths=test_masks,
    transform=val_transform
  )

  train_loader = DataLoader(
    train_ds, 
    shuffle=True,
	batch_size=batch_size, 
    pin_memory=pin_memory,
	num_workers = os.cpu_count())
  

  val_loader = DataLoader(val_ds, 
      shuffle=False,
	  batch_size = batch_size, 
      pin_memory = pin_memory,
	  num_workers = os.cpu_count())

  return train_loader, val_loader