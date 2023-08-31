import torch.nn as nn
from core.models.resnet import Resnet1D

class Encoder(nn.Module):
    def __init__(self,
                 input_emb_dim = 3,
                 output_emb_dim = None,
                 down_t = 3,
                 stride_t = 2,
                 dim = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None):
        super().__init__()
        
        blocks = []
        self.input_emb_dim = input_emb_dim
        self.output_emb_dim = output_emb_dim if output_emb_dim is not None else dim
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_dim, dim, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):
            input_dim = dim
            block = nn.Sequential(
                nn.Conv1d(input_dim, dim, filter_t, stride_t, pad_t),
                Resnet1D(dim, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(dim, self.output_emb_dim, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        
        if x.shape[1] != self.input_emb_dim:
            x = x.permute(0,2,1)
            return self.model(x).permute(0,2,1)
            
        
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self,
                 input_emb_dim = 3,
                 output_emb_dim = None,
                 down_t = 3,
                 stride_t = 2,
                 dim = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []
        
        self.input_emb_dim = input_emb_dim
        self.output_emb_dim = output_emb_dim if output_emb_dim is not None else dim

        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(self.output_emb_dim, dim, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = dim
            block = nn.Sequential(
                Resnet1D(dim, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(dim, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(dim, dim, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(dim, input_emb_dim, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        
        if x.shape[1] != self.input_emb_dim:
            x = x.permute(0,2,1)
            return self.model(x).permute(0,2,1)
            
        return self.model(x)
    
