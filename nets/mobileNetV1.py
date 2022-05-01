import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

# MobileNetv1定义如下
class MobileNetv1(nn.Module):
    def __init__(self):
        super(MobileNetv1, self).__init__()
        
        # 普通卷积
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
        # 深度可分离卷积
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
     
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )
 
        self.model = nn.Sequential(
            conv_bn(  3,  32, 2),   #3,224,224>>>32,112,112
            conv_dw( 32,  64, 1),   #32,112,112>>>64,112,112
            conv_dw( 64, 128, 2),   #64,112,112>>>128,56,56
            conv_dw(128, 128, 1),   #128,56,56>>>128,56,56
            conv_dw(128, 256, 2),   #128,56,56>>>256,28,28
            conv_dw(256, 256, 1),   #256,28,28>>>256,28,28
            conv_dw(256, 512, 2),   #256,28,28>>>512,14,14
            conv_dw(512, 512, 1),   #512,14,14>>>512,14,14
            conv_dw(512, 512, 1),   #512,14,14>>>512,14,14
            conv_dw(512, 512, 1),   #512,14,14>>>512,14,14
            conv_dw(512, 512, 1),   #512,14,14>>>512,14,14
            conv_dw(512, 512, 1),   #512,14,14>>>512,14,14
            conv_dw(512, 1024, 2),  #512,14,14>>>1024,7,7
            conv_dw(1024, 1024, 1), #1024,7,7>>>32,7,7
            nn.AvgPool2d(7),        #1024,7,7>>>1024,1,1
        )

        self.fc = nn.Linear(1024, 1000)#1024>>>1000

        features = [conv_bn( inp=3,oup=32,stride=2)]


    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024) #相当于numpy中reshape
        x = self.fc(x)
        return x

    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def Unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True

# 我们再仿照MobileNetv1的结构定义一个普通卷积构成的网络 ConvNet
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
 
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
 
        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_bn( 32,  64, 1),
            conv_bn( 64, 128, 2),
            conv_bn(128, 128, 1),
            conv_bn(128, 256, 2),
            conv_bn(256, 256, 1),
            conv_bn(256, 512, 2),
            conv_bn(512, 512, 1),
            conv_bn(512, 512, 1),
            conv_bn(512, 512, 1),
            conv_bn(512, 512, 1),
            conv_bn(512, 512, 1),
            conv_bn(512, 1024, 2),
            conv_bn(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)
 
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


if __name__ =="__main__":
    model = MobileNetv1()
    input = torch.randn(1, 3, 224, 224)

    model.fc=nn.Linear(1024,50)
    
    madds, params = profile(model, inputs=(input, ))
    output = model(input)

    print('============',type(model.model.parameters[0]))
    # for content in model.model.parameters:
    #     print(model.model.parameters)
    # print(model)
    # print('*********output size:',output.size())
    # print('*********MobileNetv1 #madds:{}, #params:{}'.format(madds, params))
    
    # model = ConvNet()
    # input = torch.randn(1, 3, 224, 224)
    # madds, params = profile(model, inputs=(input, ))
    # print('ConvNet #madds:{}, #params:{}'.format(madds, params))
    pass