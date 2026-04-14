import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np

mpi_combo=[(0,1),(1,2,),(2,6),(7,12),(12,11),(11,10),(5,4),(4,3),(3,6),(7,13),(13,14),(14,15),(6,7),(7,8),(8,9)]#комбинации суставов для комбинирования 

def get_keypoints_from_heatmaps(heatmap,treshold=0.3):
    c,h,w=heatmap.shape
    resh_heatmap=heatmap.reshape(c,-1)
    max_vals,max_idx=resh_heatmap.max(dim=1)
    pred_x=(max_idx%w).float()
    pred_y=(max_idx//w).float()

    visibility=(max_vals>treshold).float()

    pred_x,pred_y=visibility*pred_x,visibility*pred_y
    return pred_x,pred_y



def show_result(img,heatmap):
    img=img.permute(1,2,0).detach().cpu().numpy()
    full_heatmap=torch.max(heatmap,dim=0)[0].detach().cpu().numpy()

    px,py=get_keypoints_from_heatmaps(heatmap,treshold=0.2)

    #масштабирование
    scale=img.shape[0]/heatmap.shape[1]#хз
    px *=scale
    py *=scale

    figure=plt.figure(figsize=(12,4),dpi=100)

    #1
    plt.subplot(1,5,1)
    plt.title('Изображение')
    plt.imshow(img)
    #2
    plt.subplot(1,5,2)
    plt.title('Предсказание')
    plt.imshow(full_heatmap)

    #3
    plt.subplot(1,5,3)
    plt.title('Предсказание суставов')
    plt.imshow(img)
    px,py=px.cpu().numpy(),py.cpu().numpy()
    plt.scatter(px,py,c='red',s=20)
    #4
    plt.subplot(1,5,4)
    plt.title('Предсказание костей')
    plt.imshow(img)
    

    for idx in mpi_combo:
        #if idx[0]<len(px) and idx[1]<len(py):
        if px[idx[0]]==0 or px[idx[1]]==0:
            continue
        else:
            plt.plot([px[idx[0]],px[idx[1]]],[py[idx[0]],py[idx[1]]],linewidth=2,color='lime',alpha=0.7)#alpha=0.7,color='lime

    return figure
    

    for idx in mpi_combo:
        #if idx[0]<len(px) and idx[1]<len(py):
        if px[idx[0]]==0 or px[idx[1]]==0:
            continue
        else:
            plt.plot([px[idx[0]],px[idx[1]]],[py[idx[0]],py[idx[1]]],linewidth=2,color='lime',alpha=0.7)#alpha=0.7,color='lime

    return figure