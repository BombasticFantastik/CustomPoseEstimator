from Dataset import PoseDataset
from torch.utils.data import Subset,DataLoader
import yaml
from model import SkeletNet
from torch.optim import Adam
from torch.nn import MSELoss
import os
from Loop import Train
import torch

option_path='config.yaml'
with open(option_path,'r') as file_option:
    files_option=yaml.safe_load(file_option)

img_path=files_option['paths']['img']
label_path=files_option['paths']['label']
eval_lenght=files_option['eval_lenght']
device=files_option['device']

pose_data=PoseDataset(img_path=img_path,label_path=label_path,img_size=256,heatmap_size=64,sigma=2)
train_pose_data=Subset(dataset=pose_data,indices=range(len(pose_data)-eval_lenght))

train_pose_dataloader=DataLoader(dataset=train_pose_data,batch_size=16,shuffle=False,drop_last=True)

model=SkeletNet()
optimizer=Adam(model.parameters())
loss_fn=MSELoss()


#убрать прямой путь
if os.path.isfile(files_option['paths']['weights']):
    weights_dict=torch.load(files_option['paths']['weights'],weights_only=True)
    model.load_state_dict(weights_dict)
    print('Веса обнаружены')

Train(epochs=5,model=model,dataloader=train_pose_dataloader,optimizer=optimizer,loss_fn=loss_fn,device=device)

