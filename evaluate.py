import torch
import torch.nn.functional as F
from tqdm import tqdm
from post_process import iou_avg_score

from utils.dice_score import multiclass_dice_coeff, dice_coeff


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dscore = 0
    iscore = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, 
            desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        image = image.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            mask_pred = net(image)

        iscore += iou_avg_score(mask_pred, mask_true)

        # move images and labels to correct device and type
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        mask_true = mask_true.to(device=device, dtype=torch.long)
        dscore += dice_score(net.n_classes, mask_pred, mask_true)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dscore, iscore
    return dscore / num_val_batches, iscore / num_val_batches


def dice_score(num_classes, mask_pred, mask_true):
    if num_classes == 1:
        mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
        # compute the Dice score
        return dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
    else:
        mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
        # compute the Dice score, ignoring background
        return multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], 
                reduce_batch_first=False)



