from torch.utils.data import Dataset
import json
from torchvision import transforms
import os
from PIL import Image
import torch
import numpy as np

class PoseDataset(Dataset):
    def __init__(self,img_path,label_path,img_size,heatmap_size,sigma):
        super().__init__()

        self.img_size=img_size
        self.heatmap_size=heatmap_size
        self.sigma=sigma
        
        
        with open(label_path,'r') as file_option:
            self.labels=json.load(file_option)

        self.images=[os.path.join(img_path,label['img_paths']) for label in self.labels]

        self.trans=transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.images)

    def generate_heatmaps(self,joints):
        stride=self.img_size/self.heatmap_size
        h,w=self.heatmap_size,self.heatmap_size
        heatmaps=np.zeros((len(joints),h,w),dtype=np.float32)
        
        for i,joint in enumerate(joints):
            x,y,visibility=joint
            if visibility==0:
                continue
            mu_x=x/stride
            mu_y=y/stride
    
            x_range,y_range=np.arange(0,self.heatmap_size,1),np.arange(0,self.heatmap_size,1)
            xx,yy=np.meshgrid(x_range,y_range)#инвертированно
    
            d2=(xx-mu_x)**2+(yy-mu_y)**2
            exponent=d2/(2*self.sigma**2)
            heatmap=np.exp(-exponent)
    
            heatmaps[i]=heatmap
        return torch.from_numpy(heatmaps)


    def __getitem__(self,idx,augm_p=0.7):
        label=self.labels[idx]

        img=Image.open(self.images[idx])

        img_center=np.array(label['objpos'])
        img_scale=label['scale_provided']
        crop_size=img_scale*200

        #аугментация
        if random.random()<augm_p:
            offset_x=random.uniform(-0.25,0.25)*crop_size
            offset_y=random.uniform(-0.25,0.25)*crop_size
            img_center[0]+=offset_x
            img_center[1]+=offset_y

            crop_size=crop_size*random.uniform(0.5,1.2)

        #конец аугментации
        
        left=img_center[0]-crop_size/2
        top=img_center[1]-crop_size/2
        right=left+crop_size
        bottom=top+crop_size

        img_crop=img.crop((left,top,right,bottom))
        #orig_w,orig_h=img.size
        #tensor_img=self.trans(img)
        
        

        joints=np.array(label['joint_self'])
        joints[:,0]-=left
        joints[:,1]-=top

        #убираем не попавшие суставы

        for i in range(len(joints)):
                if (joints[i, 0] < 0 or joints[i, 0] >= crop_size or 
                    joints[i, 1] < 0 or joints[i, 1] >= crop_size):
                    joints[i, 2] = 0.0

        #заканчиваем убирать не попавшие суставы

        final_w,final_h=img_crop.size
        tensor_img=self.trans(img_crop)
        
        joints[:,0]=joints[:,0]*(self.img_size/final_w)
        joints[:,1]=joints[:,1]*(self.img_size/final_h)

        target=self.generate_heatmaps(joints=joints)

        
        return {
            'img':tensor_img,
            'label':target
        }