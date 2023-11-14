## CATFA-Net-pytorch
Official Pytorch code of "CATFA-Net: An Inspired Trans-Convolutional Network for Medical Image Segmentation" - IEEE TMI

## Usage
Train RFAUCNxt using a medical image dataset containing binary labels. For multi-class segmentation, the training script "train.py" needs to be modified.

# 1. Install Dependencies

Please prepare an environment with python = 3.10.x, and then use the command 
```bash 
pip install -r requirements.txt
```
for installing the dependencies. Review and check the versions of the libraries before installing. If you are running some other vesion of
python, be sure to use the correct versions of the libraries.

# 2. Run train.py in Terminal
Arguments to be parsed from the terminal:

1. dir : Directory of the dataset. Inside the dataset directory, the data should be organized in the following manner:
   
   ```raw
   a. '/train' : should contain training images.
   
   b. '/train_masks' : should contain corresponding binary masks of the training images.

   c. '/test' : should contain images for testing every epoch.

   d. '/test_masks' : should contain corresponding binary masks of the test images.
   ```

2. --pre_split: Boolean "True" or "False" values are taken. Mentions whether a dataset has predefined split or not.
      
3. --split: If the dataset is not pre-split, mention the test split ratio. This value defaults to 0.2 .
      
4. --result_dir: Path to result directory where all the sample validation outputs, segmentation report and model weights will be stored after training.

5. --save: Boolean "True" or "False" values are taken. If "True" then the training loop will save the model checkpoint in every epoch.

6. --save_file_name: Name of the .pth.tar file where the model weights will be saved.

7. --num_epochs: Total number of training epochs. Needs to be an integer value.

8. --lr: Learning Rate of the optimizer. Might be a floating point value.

9. --batch_size: Batch size of train and validation data per training epoch.

10. --model_size: Size of CATFA-Net model. Takes values such as 'small' or 'large'.

11. --num_workers: Number of CPU workers for DataLoader object.

12. --pin_mem: Boolean "True" or "False" values are taken. If True, the dataloader will be pinned to memory.

13. --optimizer: Choice of Optimizer function. Values can be 'SGD', 'Adam' or 'AdamW'. We use 'AdamW' as a choice of optimizer and Dice Loss as a loss function of choice.

14. --pretrained: Boolean "True" or "False" values are taken. If True, CATFA-Net is loaded with a pre-trained ConvNext encoder backbone.

15. --save_roc_pr: Boolean "True" or "False" values are taken and defaults to "False". If "True", two sheet containing information to plot ROC and PR curves will be generated respectively, which can be used   
                      to plot the corresponding curves in Origin.
  

An example bash command for training CATFA-Net (this will load the small variant) would be:
```bash
python3 train.py /path/to/dataset
--pre_split False --split 0.25
--result_dir results --save True
--save_file_name results/catfanet_small.pth.tar --num_epochs 50
--lr 1e-4 --batch_size 8
--model_size small --num_workers 2 --pin_mem True
--optimizer AdamW --pretrained True
--save_roc_pr False
```
