import torch.nn as nn

'''
Contains the classes composing ResNet for TSC from 
https://arxiv.org/pdf/1611.06455.pdf

Initialize the model with model = ResNetTSC()
'''


class BasicBlock(nn.Module):
    # Basic CNN block for use in the ResNet Block
    def __init__(self, channels, kernel_size):
        super(BasicBlock, self).__init__()

        self.conv  = nn.Conv1d(in_channels = channels, out_channels = channels, \
                               kernel_size = kernel_size, padding='same')
        self.bn    = nn.BatchNorm1d(channels)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        return x

class ResNetBlock(nn.Module):
    # creates a residual block for use in the ResNetTSC architecture
    def __init__(self, channels, num_blocks, kernel_size):
        super(ResNetBlock, self).__init__()
        blocks = []
        for _ in range(num_blocks):
            blocks.append(BasicBlock(channels, kernel_size))

        self.resnet_block = nn.Sequential(*blocks)
        self.activ = nn.ReLU()
        

    def forward(self, x):
        # pass input through resnet block
        out = self.resnet_block(x)
        # add x back after residual block
        y = out + x
        # pass residual output through ReLU
        h = self.activ(y)
        return h

class ResNetTSC(nn.Module):
    """
    DESC
    ---
    Initializes ResNetTSC Class
    ResNetTSC class implements architecture from 
    https://arxiv.org/pdf/1611.06455.pdf
    ---
    INPUTS
    ---
    num_classes (int): the number of classes desired for the output
    in_channels (int): number of feature channels in the input
    ---
    RETURN
    ---
    ResNetTSC Model Instance
    """

    def __init__(self, num_classes=2, in_channels=10):
        super(ResNetTSC, self).__init__()

        # Configurations of Stages 
        self.stage_cfgs = [
            # channels, # blocks, kernel size
            [64, 3, 8],
            [128, 3, 5],
            [128, 3, 3],

        ]
        self.num_classes=num_classes

        # build layers
        layers = []
        for stage_idx, curr_stage in enumerate(self.stage_cfgs):
            num_channels, num_blocks, kernel_size = curr_stage
            # upsample the channels to the correct stage size
            layers.append(nn.Conv1d(in_channels=in_channels, \
                out_channels=self.stage_cfgs[stage_idx][0], kernel_size=1))
            # append resnet block for this stage
            layers.append(ResNetBlock(num_channels, num_blocks, kernel_size))
            # set in_channels for next stage
            in_channels = num_channels

        # Pool and Flatten      
        layers.append(nn.AdaptiveAvgPool2d((self.num_classes,1)))
        layers.append(nn.Flatten())

        # create sequential for use in forward
        self.layers = nn.Sequential(*layers) 
                   
    def forward(self, x):
        # data given as (batch size, sequence length, feature channels)
        # permute to (batch size, feature channels, sequence length) for CNNs
        x = x.permute((0, 2, 1))
        out = self.layers(x)
        return out
    