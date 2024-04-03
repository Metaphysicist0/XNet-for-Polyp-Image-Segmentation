

import torch
from tqdm import tqdm
import numpy as np
import os
from models.UNet import UNET
from dataset import PolypDataset
from torch.utils import data
from metrics import dice_coef
from PIL import Image


image_path = r"./Data/TestDataset/images/"
mask_path = r"./Data/TestDataset/masks/"
image_all = [image_path + f for f in os.listdir(image_path) if f.endswith('.jpg') or f.endswith('.png')]
mask_all =  [mask_path + f for f in os.listdir(mask_path) if f.endswith('.jpg') or f.endswith('.png')]
test_dataset = PolypDataset(image_all,mask_all,224)
test_dataloader = data.DataLoader(test_dataset,batch_size=1)


model = UNET(n_classes=1)
model.load_state_dict(torch.load('./models/models.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

save_path = "./results/"
dice = 0
model.eval()
with torch.no_grad():
    for i, (image, gt) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        # image, gt, name = test_loader.load_data()
        # name = images[index].split('/')[-1]
        # name = str(images[1 * i: 1 * (i + 1)]).split('/')
        name = str(image_all[1 * i: 1 * (i + 1)]).split('/')[-1]
        name = name.split('.')[0] + '.png'
        # if name.endswith('.jpg'):
        #     name = name.split('.jpg')[0] + '.png'
        # gt.float()
        # gt = gt.view(-1,gt.shape[2],gt.shape[3])
        # gt = torch.squeeze(gt, dim=0)
        # gt = np.asarray(gt, np.float32)
        # gt /= (gt.max() + 1e-8)
        if torch.cuda.is_available():
            image = image.cuda()
            gt = gt.cuda()

        gt = gt.squeeze()
        gt = gt.data.cpu().numpy().astype(np.uint8)
        mask = np.zeros([gt.shape[-2],gt.shape[-1]],dtype=np.uint8)
        output = model(image)
        output = output.squeeze()

        output = output.sigmoid().data.cpu().numpy()
        # output = (output - output.min()) / (output.max() - output.min() + 1e-8)
        for idh in range(output.shape[-2]):
            for idw in range(output.shape[-1]):
                if output[idh,idw] > 0.5:
                    mask[idh,idw] = 1

        dice = dice_coef(mask, gt) + dice

        mask[mask == 1] = 255
        # misc.imwrite(save_path+name, res)
        # plt.savefig(save_path+name, res)
        output = output.astype(np.uint8)
        mask = Image.fromarray(mask)

        mask.save(save_path + name)


dice_mean = dice/len(image_all)
print(dice_mean)
