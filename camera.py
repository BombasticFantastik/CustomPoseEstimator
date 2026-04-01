import cv2
from torchvision import transforms
import torch 
from model import SkeletNet
from visualization import show_result
import matplotlib.pyplot as plt
import numpy as np
def figure_to_array(fig):
    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()
    img_array = np.array(rgba_buffer)
    return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

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
        figure=show_result(tensor_frame,pred[0])

    plot_img = figure_to_array(figure)
    cv2.imshow('Matplotlib in OpenCV', plot_img)
    

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()