import torch
from tqdm import tqdm

def Train(epochs,model,dataloader,optimizer,loss_fn,device):
    for epoch in range(epochs):
        model.train()
        model=model.to(device)
        for batch in (pbar:=tqdm(dataloader)):
            optimizer.zero_grad()
            pred=model(batch['img'].to(device))
           
            loss=loss_fn(pred,batch['label'].to(device))
            loss.backward()
            loss_item=loss.item()
    
            pbar.set_description(f'{loss_item}')
            optimizer.step()
        torch.save(model.state_dict(),'/home/artemybombastic/MyGit/SkeletonNet/CustomPoseEstimatorData/weights.pth')