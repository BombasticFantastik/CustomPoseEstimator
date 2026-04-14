import numpy as np
from visualization import get_keypoints_from_heatmaps



def PCK_accuracy(img,pred_heatmap,label_heatmap):
    img=img.permute(1,2,0).detach().cpu().numpy()
    pred_px,pred_py=get_keypoints_from_heatmaps(pred_heatmap,treshold=0.2)
    label_px,label_py=get_keypoints_from_heatmaps(label_heatmap,treshold=0.2)
    pred=np.array([pred_px.cpu(),pred_py.cpu()])
    label=np.array([label_px.cpu(),label_py.cpu()])
    accuracy=(((abs(pred[0]-label[0])+abs(pred[1]-label[1]))<5).sum()/len(pred[0])).item()
    return accuracy

def test_accuracy(model,test_pose_dataloader,device='cpu'):
    full_accuracy=[]
    for batch in test_pose_dataloader:
        pred=model(batch['img'].to(device))
        
        for i in range(pred.size(0)):
            full_accuracy.append(PCK_accuracy(batch['img'][i],pred[i],batch['label'][i]))
        return sum(full_accuracy)/len(full_accuracy)