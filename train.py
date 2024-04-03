import torch
from tqdm import tqdm
from torch import optim
import os
from models.UNet import UNET
from sklearn.model_selection import train_test_split
from dataset import PolypDataset
from torch.utils import data
import losses
import pandas as pd
from metrics import iou_score
from collections import OrderedDict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader,model, optimizer,criterion):
    model.train()
    losses = AverageMeter()
    ious = AverageMeter()
    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        output = model(input)
        loss = criterion(output, target)
        iou = iou_score(output, target)

        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])
    return log


def validate(val_loader,model,criterion):
    model.eval()
    losses = AverageMeter()
    ious = AverageMeter()
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))

        log = OrderedDict([
            ('loss', losses.avg),
            ('iou', ious.avg),
        ])

    return log
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNET(n_classes=1).to(device)

criterion = losses.BCEDiceLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

log = pd.DataFrame(index=[], columns=[
    'epoch', 'loss', 'iou', 'val_loss', 'val_iou'
])


image_path = r"./Data/TrainDataset/images/"
mask_path = r"./Data/TrainDataset/masks/"

image_all = [image_path + f for f in os.listdir(image_path) if f.endswith('.jpg') or f.endswith('.png')]
mask_all =  [mask_path + f for f in os.listdir(mask_path) if f.endswith('.jpg') or f.endswith('.png')]

train_im_path, val_im_path, train_mask_path,val_mask_path = train_test_split(image_all,mask_all,test_size=0.1,random_state=101)
train_dataset = PolypDataset(train_im_path,train_mask_path,224)
val_dataset = PolypDataset(val_im_path,val_mask_path,224)

train_dataloader = data.DataLoader(train_dataset,batch_size=16,shuffle=True,pin_memory=True,drop_last=False)
val_dataloader = data.DataLoader(val_dataset,batch_size=16,pin_memory=True,drop_last=False)

best_iou = 0
epochs = 100
trigger = 0
for epoch in range(epochs):
    print('Epoch [%d/%d]' %(epoch, epochs))

    # train for one epoch
    train_log = train(train_dataloader,model,optimizer,criterion)
    # evaluate on validation set
    val_log = validate(val_dataloader, model, criterion)

    print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
        %(train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

    tmp = pd.Series([
            int(epoch),
            train_log['loss'],
            train_log['iou'],
            val_log['loss'],
            val_log['iou'],
        ], index=['epoch', 'loss', 'iou', 'val_loss', 'val_iou'])

    log = log.append(tmp, ignore_index=True)
    log.to_csv('models/log.csv', index=False)

    trigger += 1

    if val_log['iou'] > best_iou:
        torch.save(model.state_dict(), 'models/models.pth')
        best_iou = val_log['iou']
        print("=> saved best models")
        trigger = 0

    torch.cuda.empty_cache()
