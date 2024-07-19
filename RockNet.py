import torch
import torch.nn as nn

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.bn(self.conv(x))

class Backbone(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Backbone, self).__init__()
        self.silu = nn.SiLU(inplace=True)
        '''branch1'''
        self.branch1_1 = ConvBN(in_channels, in_channels, 3, stride)
        self.branch1_2 = ConvBN(in_channels, out_channels, 1, 1)
        '''branch2'''
        self.branch2_1 = ConvBN(in_channels, out_channels, 3, stride)
        '''branch3'''
        self.branch3_1 = ConvBN(in_channels, out_channels, 5, stride)
        '''branch4'''
        self.branch4_2 = ConvBN(out_channels, out_channels, 3, 1)

        '''torch.add'''
        self.bn_sum = nn.BatchNorm2d(out_channels)
        '''maxpool2d'''
        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)        

    def forward(self, x):
        '''branch1'''
        branch1 = self.branch1_1(x)
        branch1 = self.silu(branch1)
        branch1 = self.branch1_2(branch1)
        '''branch2'''
        branch2 = self.branch2_1(x)
        '''branch3'''
        branch3 = self.branch3_1(x)
        '''branch4'''
        branch4 = self.silu(branch3)
        branch4 = self.branch4_2(branch4)
        
        '''add'''
        out = branch1+branch2+branch3+branch4
        '''maxpool'''
        out = self.maxpool2d(out)
        out = self.bn_sum(out)
        out = self.silu(out)
        
        return out
    
# Define the RockNet model
class RockNet(nn.Module):
    def __init__(self, num_classes=1000):
        self.base_block = Backbone
        super(RockNet, self).__init__()
        self.silu = nn.SiLU(inplace=True)       # Define the activation function
        # The first CBS model                     input: B*3*224*224 -> output: B*32*224*224
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        # Backbone part                                                             # the size of feature map
        self.layer1 = self.base_block(in_channels=32, out_channels=64, stride=2)    # 112*112
        self.layer2 = self.base_block(in_channels=64, out_channels=128, stride=2)   # 56*56
        self.layer3 = self.base_block(in_channels=128, out_channels=256, stride=2)  # 28*28
        self.layer4 = self.base_block(in_channels=256, out_channels=512, stride=2)  # 14*14
        self.layer5 = self.base_block(in_channels=512, out_channels=1024, stride=2) # 7*7

        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.fc = nn.Linear(1024,num_classes)

    def forward(self, x): 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.silu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
