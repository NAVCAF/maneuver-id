import torch
from torch import nn

def enable_dropout(model):
    '''
    DESC
    ---
    This function sets the dropout layers to train mode
    ---
    INPUTS
    ---

    model: the model
    ---
    OUTPUTS
    ---
    None

    '''
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

class bcnn_block(nn.Module):
    '''
    DESC
    ---
    A conv-act-dropout-pool block with MC dropout
    ---
    INPUTS
    ---
    in_channels: input channels
    out_channels: output_channels
    kernel_size: kernel size
    p: dropout rate

    '''
    def __init__(self, in_channels, out_channels, kernel_size, p):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.Dropout = nn.Dropout(p)
        self.pool = nn.MaxPool1d(3)
    
    def forward(self, x, final = False):
        out = self.conv(x)
        out = self.relu(out)
        out = self.Dropout(out)
        if not final:
            out = self.pool(out)

        return out

class Bayes_CNN(nn.Module):
    '''
    DESC
    ---
    A Bayes CNN module
    ---
    INPUTS
    ---
    in_channels: input channels
    out_channels_list: list of output_channels
    kernel_sizees: list of kernel sizes
    dropout_probs: lits of dropout rates

    '''
    def __init__(self, in_channels = 9, out_channel_list = [128, 256, 256, 128], 
                 p = 0.5, kernel_sizes = [7,5,3,2]):
        super().__init__()
        in_channel_list = [in_channels] + out_channel_list[:-1]
        self.blocks = nn.ModuleList([
            bcnn_block(in_channel_list[i], out_channel_list[i], kernel_sizes[i], p)
            for i in range(len(out_channel_list))
        ])
        self.final_pool = nn.AdaptiveMaxPool1d(1)

        self.linear= nn.Linear(128,2)
    
    def forward(self, x):

        out = torch.permute(x, (0,2,1))
        num_blocks = len(self.blocks)
        for i,module in enumerate(self.blocks):
            if i < num_blocks-1:
                out = module(out)   
            else:
                out = module(out, final = True)     
        
        out = self.final_pool(out).squeeze()

        out = self.linear(out)

        return out