from torch import nn 

class FCNBlock(nn.Module):
    
    # A single block used in the FCN network
    def __init__(self, kernel_size = 3, in_channels = 128, out_channels = 128):

        """
        DESC
        ---
        Initializes a block in the FCN layer consisting of a convolution
        a batch norm and a ReLU activation
        ---
        INPUTS
        ---
        kernel_size (int, optional): the kernel width to the convolutional layer
        in_channels (int, optional): number of channels in input
        out_channels (int, optional): number of channels in the output
        ---
        RETURN
        ---
         FCNBlock instance
        """
        
        super(FCNBlock, self).__init__()

        # set up block
        self.conv  = nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size)
        self.bn    = nn.BatchNorm1d(out_channels)
        self.activ = nn.ReLU()

    def forward(self, x):

        # pass through conv -> bn -> relu
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        return x

class FullyConvNetwork(nn.Module):
    def __init__(self, input_dims = 10, num_classes = 2):

        """
        DESC
        ---
        Initializes a fully convolutional network (FCN)
        ---
        INPUTS
        ---
        input_dims (int, optional): number of channels in input time series
        num_classes (int, optional): number of classes
        ---
        RETURN
        ---
            FullyConvNetwork instance
        """

        super(FullyConvNetwork, self).__init__()
        self.num_classes = num_classes

        self.blocks = nn.Sequential(
            FCNBlock(in_channels = input_dims, out_channels = 128, kernel_size = 8),
            FCNBlock(in_channels = 128, out_channels = 256, kernel_size = 5),
            FCNBlock(in_channels = 256, out_channels = 128, kernel_size = 3),
            nn.AdaptiveAvgPool2d((num_classes,1)),
            nn.Flatten()
        )


    def forward(self, x):

        # permute to (batch_size, channels, time)
        x = x.permute((0, 2, 1))
        out = self.blocks(x)
        return out