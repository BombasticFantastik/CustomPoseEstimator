from Dataset import PoseDataset
from torch.utils.data import Subset,DataLoader
import yaml
from model import SkeletNet
from torch.optim import Adam
from torch.nn import MSELoss
import os
from Loop import Train
import torch
from tqdm import tqdm
from metric import PCK_accuracy

option_path='config.yaml'
with open(option_path,'r') as file_option:
    files_option=yaml.safe_load(file_option)

img_path=files_option['paths']['img']
label_path=files_option['paths']['label']
eval_lenght=files_option['eval_lenght']
device=files_option['device']

pose_data=PoseDataset(img_path=img_path,label_path=label_path,img_size=256,heatmap_size=64,sigma=2)
test_pose_data=Subset(dataset=pose_data,indices=range(len(pose_data)-eval_lenght,len(pose_data)))
test_pose_dataloader=DataLoader(dataset=test_pose_data,batch_size=16,shuffle=True,drop_last=True)

model=SkeletNet().to(device)

if os.path.isfile(files_option['paths']['weights']):
    weights_dict=torch.load(files_option['paths']['weights'],weights_only=True)
    model.load_state_dict(weights_dict)
    print('Веса обнаружены')

#оцениваем качество
full_accuracy=[]
for batch in (pbar:=tqdm(test_pose_dataloader)):
    pred=model(batch['img'].to(device))
    for i in range(16):
        full_accuracy.append(PCK_accuracy(batch['img'][i],pred[i],batch['label'][i]))
print(sum(full_accuracy)/len(full_accuracy))
