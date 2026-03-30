import cv2
from torchvision import transforms
import torch 
from model import SkeletNet
from visualization import img2skelet

model=SkeletNet().eval()
weights_dict=torch.load('/home/artemybombastic/MyGit/SkeletonNet/CustomPoseEstimatorData/weight.pth',weights_only=True)
model.load_state_dict(weights_dict)


trans=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])



camera=cv2.VideoCapture(-1)

while camera.isOpened():

    sucess,frame=camera.read()

    tensor_frame=trans(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        pred=model(tensor_frame.unsqueeze(0))
        pred=img2skelet(tensor_frame,pred[0])
        


    result = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    cv2.imshow('some',result)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()