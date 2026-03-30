import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np

mpi_combo=[(0,1),(1,2,),(2,6),(7,12),(12,11),(11,10),(5,4),(4,3),(3,6),(7,13),(13,14),(14,15),(6,7),(7,8),(8,9)]#комбинации суставов для комбинирования 

def get_keypoints_from_heatmaps(heatmap):
    c,h,w=heatmap.shape
    resh_heatmap=heatmap.reshape(c,-1)
    max_idx=resh_heatmap.argmax(dim=1)
    pred_x=(max_idx%w).float()
    pred_y=(max_idx//w).float()
    return pred_x,pred_y


# def img2skelet(img,heatmap):
#     img=cv2.cvtColor(img.permute(1,2,0).detach().cpu().numpy(),cv2.COLOR_RGB2BGR)
#     if img.max() <= 1.0:
#         img = (img * 255).astype(np.uint8)
#     else:
#         img = img.astype(np.uint8)


#     full_heatmap=torch.max(heatmap,dim=0)[0].detach().cpu().numpy()

#     px,py=get_keypoints_from_heatmaps(heatmap)

#     #масштабирование
#     scale=img.shape[0]/heatmap.shape[1]#хз
#     px *=scale
#     py *=scale

#     px,py=px.cpu().numpy(),py.cpu().numpy()

    
#     for idx in mpi_combo:

#         pt1=(int(px[idx[0]]),int(py[idx[0]]))
#         pt2=(int(py[idx[1]]),int(py[idx[1]]))
#         if px[idx[0]]==0 or px[idx[1]]==0:
#             continue
#         else:
#             cv2.line(img,pt1,pt2,color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

#     return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def img2skelet(img,heatmap):
    img=cv2.cvtColor(img.permute(1,2,0).detach().cpu().numpy(),cv2.COLOR_RGB2BGR)
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)


    full_heatmap=torch.max(heatmap,dim=0)[0].detach().cpu().numpy()

    px,py=get_keypoints_from_heatmaps(heatmap)

    #масштабирование
    scale=img.shape[0]/heatmap.shape[1]#хз
    px *=scale
    py *=scale

    px,py=px.cpu().numpy(),py.cpu().numpy()

    
    for idx in mpi_combo:

        pt1=(int(px[idx[0]]),int(py[idx[0]]))
        pt2=(int(py[idx[1]]),int(py[idx[1]]))
        if px[idx[0]]==0 or px[idx[1]]==0:
            continue
        else:
            cv2.line(img,pt1,pt2,color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)