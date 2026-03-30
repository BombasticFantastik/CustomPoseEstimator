from torchvision.models import resnet18,ResNet18_Weights
from torch import nn 
from torch.nn import Module

class SkeletNet(Module):
    def __init__(self):
        super().__init__()
        self.cnn=resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn=nn.Sequential(
            *list(self.cnn.children())[:-2]
        )
        
        self.dnn=nn.Sequential(
            nn.ConvTranspose2d(512,256,kernel_size=(4),stride=2,padding=1),#увеличивает размер в 2 раза
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256,256,kernel_size=(4),stride=2,padding=1),#увеличивает размер в 2 раза
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256,256,kernel_size=(4),stride=2,padding=1),#увеличивает размер в 2 раза
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        for lay in self.dnn:
            if isinstance(lay,nn.ConvTranspose2d):
                nn.init.normal_(lay.weight, std=0.001)
            # elif isinstance(lay,nn.BatchNorm2d):
            #     nn.init.constant_(lay.bias,0)

        
        self.predictor=nn.Sequential(
            nn.Conv2d(256,16,kernel_size=1)
        )

        for lay in self.predictor:
            if isinstance(lay,nn.ConvTranspose2d):
                nn.init.normal_(lay.weight, std=0.001)
        
    def forward(self,x):
        out=self.cnn(x)
        out=self.dnn(out)
        out=self.predictor(out)
        return out