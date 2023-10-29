"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import collections

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block 
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, name, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            collections.OrderedDict([
                (name + "Conv2d", nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
                (name + "BatchNorm2d" ,nn.BatchNorm2d(out_channels)),
                (name + "ReLU",nn.ReLU(inplace=True)),
                (name + "Conv2d2",nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False)),
                (name + "BatchNorm2d2", nn.BatchNorm2d(out_channels * BasicBlock.expansion))
            ])
            
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    
class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=10, plus = False):
        super().__init__()

        self.in_channels = 64
        self.num_classes = num_classes
        self.last_blocks = []

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, "conv2_x")
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, "conv3_x")
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, "conv4_x")
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, "conv5_x")
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # class number of last blocks.
        if plus == True:
            for i in range(num_classes):
                self.last_blocks.append(self._make_layer(block, 512, num_block[3], 2, "conv5_" + str(i) + "_x", last=True).cuda())
            self.last_blocks = nn.ModuleList(self.last_blocks)
            self.fc = nn.Linear( num_classes * 512 * block.expansion, num_classes)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        return output
 
    def _make_layer(self, block, out_channels, num_blocks, stride, name, last = False):
        """make resnet layers(by layer i didnt mean this 'layer' was the 
        same as a neuron netowork layer, ex. conv layer), one layer may 
        contain more than one residual block 

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block 
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        if last == True:
            self.in_channels = 256
        for stride in strides:
            layers.append(block(name, self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)

    def forward(self, x, hook_layer=False, parallel_last = True):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output) # BS x 256 x 8 x 8
        
        if parallel_last == False:
            output = self.conv5_x(output) ## 이 부분에서 블럭이 parallel하게 있어야 댐.
            output = self.avg_pool(output) 
        elif parallel_last == True: # class 갯수만큼의 BS x 256 x 4 x 4
            outputs = []
            concat = self.last_blocks[0](output)
            for i in range(1, self.num_classes):
                concat = torch.cat([concat, self.last_blocks[i](output)], dim=1) # BS x class_num * 512 x 4 x 4
            output = self.avg_pool(concat)# 32 x class_num x 1 x 1
        
        if hook_layer:
            h = output.register_hook(self.activations_hook)

        
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output 

def resnet18(num_classes, plus):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, plus)

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])
