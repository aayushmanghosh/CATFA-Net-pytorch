from torch import nn
import torch
from torchmetrics import MatthewsCorrCoef, JaccardIndex, CohenKappa, Recall, Precision, ROC, Specificity, PrecisionRecallCurve
from collections import defaultdict
from tqdm import tqdm
import torchvision


metric = defaultdict(list)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

def get_metrics_data():
  return metric

def save_checkpoint(state, filename = "refuge_rfaucnxt_tiny.pth.tar"):
  print("=> Saving Checkpoint")
  torch.save(state,filename)

def load_checkpoint(checkpoint, model):
  print("=> Loading Checkpoint")
  model.load_state_dict(checkpoint["state_dict"])


def compute_roc(loader, model, device = "cuda"):
  fpr,tpr,thresholds = 0,0,0
  with torch.no_grad():
    for x,y in loader:
      x = x.to(device)
      y_in = y.to(device).type(torch.cuda.IntTensor)
      y = y.to(device).unsqueeze(1)
      predictions = model(x)

      roc = ROC(task = "binary").to(device)
      fpr, tpr, thresholds = roc(predictions, y_in.unsqueeze(1))

  return fpr.detach().cpu().numpy(), tpr.detach().cpu().numpy(),thresholds.detach().cpu().numpy()


def compute_prcurve(loader, model, device = "cuda"):
  prec, rec, thresh = 0,0,0
  with torch.no_grad():
    for x,y in loader:
      x = x.to(device)
      y_in = y.to(device).type(torch.cuda.IntTensor)
      y = y.to(device).unsqueeze(1)
      predictions = model(x)

      pr_curve = PrecisionRecallCurve(task="binary")
      prec, rec, thresh = pr_curve(predictions, y_in.unsqueeze(1))

  return prec.detach().cpu().numpy(), rec.detach().cpu().numpy(),thresh.detach().cpu().numpy()

def save_predictions_as_imgs(
    loader, model, folder = "/content/gdrive/MyDrive/results/", device = "cuda"
):

  model.eval()
  for idx, (x,y) in enumerate(loader):
    x = x.to(device = device)
    with torch.no_grad():
      preds = torch.sigmoid(model(x))
      preds = (preds>0.5).float()

    torchvision.utils.save_image(
        preds, f"{folder}/pred_{idx}.png"
    )
    torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")

  model.train()

def train_one_epoch(loader, model, optimizer, loss_fn, scaler):
  loop = tqdm(loader)
  train_loss = []
  for _, (data, targets) in enumerate(loop):
    data = data.to(device=DEVICE)
    targets = targets.float().unsqueeze(1).to(device = DEVICE)

    #forward
    with torch.cuda.amp.autocast():
      predictions = model(data)
      loss = loss_fn(predictions, targets)

    #backward
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    train_loss.append(loss.item())
    loop.set_postfix(loss = loss.item())

  return sum(train_loss)/len(train_loss)

def test_one_epoch(loader, model, train_loss, loss_fn, device = "cuda"):
  num_correct = 0
  num_pixels = 0
  dice_score = 0
  jaccard_score = 0
  cohkappa = 0
  mcc = 0
  val_loss = 0
  recall = 0
  precision = 0
  spec = 0
  model.eval()

  with torch.no_grad():
    for x,y in loader:
      x = x.to(device)
      y_in = y.to(device).type(torch.cuda.IntTensor)
      y = y.to(device).unsqueeze(1)
      predictions = model(x)
      preds1 = torch.sigmoid(predictions)
      preds = (preds1 > 0.5).float()
      val_loss += loss_fn(predictions, y)

      num_correct += (preds == y).sum()
      num_pixels += torch.numel(preds)
      dice_score += (2 * (preds * y).sum())/(
          (preds + y).sum() + 1e-8
      )
      jaccard = JaccardIndex(task='binary').to(device)
      jaccard_score += jaccard(preds, y_in.unsqueeze(1))

      coh = CohenKappa(task = 'binary').to(device)
      cohkappa += coh(preds, y_in.unsqueeze(1))

      mat = MatthewsCorrCoef(task = 'binary').to(device)
      mcc += mat(preds, y_in.unsqueeze(1))
      

      sens = Recall(task = "binary").to(device)
      recall += sens(preds, y_in.unsqueeze(1))

      prec = Precision(task = "binary").to(device)
      precision += prec(preds, y_in.unsqueeze(1))

      Spec = Specificity(task = "binary").to(device)
      spec+= Spec(preds, y_in.unsqueeze(1))

  metric["Train_loss"].append(train_loss)
  metric["Val_loss"].append((val_loss/len(loader)).detach().cpu().numpy().item())
  metric["Pixel_Wise_Acc"].append((num_correct/num_pixels*100).detach().cpu().numpy().item())
  metric["Dice_Score"].append((dice_score/len(loader)).detach().cpu().numpy().item())
  metric["Jaccard_Score"].append((jaccard_score/len(loader)).detach().cpu().numpy().item())
  metric["Precision"].append((precision/len(loader)).detach().cpu().numpy().item())
  metric["Recall"].append((recall/len(loader)).detach().cpu().numpy().item())
  metric["Specificity"].append((spec/len(loader)).detach().cpu().numpy().item())
  metric["MCC"].append((mcc/len(loader)).detach().cpu().numpy().item())
  metric["CohenKappa"].append((cohkappa/len(loader)).detach().cpu().numpy().item())

  print(
      f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
  )

  print(f"Train Loss: {train_loss}")
  print(f"Val Loss: {val_loss/len(loader)}")
  print(f"Dice Score: {dice_score/len(loader)}")
  print(f"Jaccard Score: {jaccard_score/len(loader)}")
  print(f"Precision: {precision/len(loader)}")
  print(f"Recall: {recall/len(loader)}")
  print(f"Specificity: {spec/len(loader)}")
  print(f"MCC: {mcc/len(loader)}")
  print(f"Cohen Kappa: {cohkappa/len(loader)}")


  model.train()

    
