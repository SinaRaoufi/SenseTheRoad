import torch.nn as nn
from torchvision import models

deeplab = models.segmentation.deeplabv3_resnet50(pretrained=False, 
                                                progress=True, 
                                                num_classes=2)

class DeepLabv3(nn.Module):
    def __init__(self):
        super(DeepLabv3,self).__init__()
        self.dl = deeplab
        
    def forward(self, x):
        out = self.dl(x)['out']
        return out