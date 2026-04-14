import torch
from tqdm import tqdm
from metric import test_accuracy

def Train(epochs,model,dataloader,optimizer,loss_fn,device,weights_path,test_dataloader=None):
    accuracy=0
    for epoch in range(epochs):
        model.train()
        #model=model.to(device)
        for batch in (pbar:=tqdm(dataloader)):
            optimizer.zero_grad()
            pred=model(batch['img'].to(device))
           
            loss=loss_fn(pred,batch['label'].to(device))
            loss.backward()
            loss_item=loss.item()
    
            pbar.set_description(f'{epoch}) loss:{loss_item} accuracy:{accuracy}')
            optimizer.step()


        if test_dataloader:
            accuracy=test_accuracy(model=model,test_pose_dataloader=test_dataloader,device=device)
        
        torch.save(model.state_dict(),weights_path)