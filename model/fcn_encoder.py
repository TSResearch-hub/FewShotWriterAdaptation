import torch
import torch.nn as nn
from torch.nn import Module, InstanceNorm2d, Dropout, Dropout2d, ReLU, Conv2d
from torch.nn.functional import pad
import random
import math
import torch.nn.functional as F


class FCN_Encoder(nn.Module):
    def __init__(self, output_dim, init_dropout_rate):
        super(FCN_Encoder, self).__init__()

        self.dropout = init_dropout_rate
        self.positional_encoding = SinusoidalPositionalEncoding2D(output_dim)
        
        if output_dim == 256:
            self.init_blocks = torch.nn.Sequential(*[
                ConvBlock(3, 16, stride=(1, 1), dropout=self.dropout),
                ConvBlock(16, 32, stride=(2, 2), dropout=self.dropout),
                ConvBlock(32, 64, stride=(2, 2), dropout=self.dropout),
                ConvBlock(64, 128, stride=(2, 2), dropout=self.dropout),
                ConvBlock(128, 128, stride=(2, 1), dropout=self.dropout),
                ConvBlock(128, 128, stride=(2, 1), dropout=self.dropout),
            ])
            self.blocks = torch.nn.Sequential(*[
            # self.blocks = ModuleList([
                DSCBlock(128, 128, stride=(1, 1), dropout=self.dropout),
                DSCBlock(128, 128, stride=(1, 1), dropout=self.dropout),
                DSCBlock(128, 128, stride=(1, 1), dropout=self.dropout),
                DSCBlock(128, output_dim, stride=(1, 1), dropout=self.dropout),
            ])
        elif output_dim == 1024:
            self.init_blocks = torch.nn.Sequential(*[
                ConvBlock(3, 32, stride=(1, 1), dropout=self.dropout),
                ConvBlock(32, 64, stride=(2, 2), dropout=self.dropout),
                ConvBlock(64, 128, stride=(2, 2), dropout=self.dropout),
                ConvBlock(128, 256, stride=(2, 2), dropout=self.dropout),
                ConvBlock(256, 256, stride=(2, 1), dropout=self.dropout),
                ConvBlock(256, 256, stride=(2, 1), dropout=self.dropout),
            ])
            self.blocks = torch.nn.Sequential(*[
            # self.blocks = ModuleList([
                DSCBlock(256, 512, stride=(1, 1), dropout=self.dropout),
                DSCBlock(512, 512, stride=(1, 1), dropout=self.dropout),
                DSCBlock(512, 512, stride=(1, 1), dropout=self.dropout),
                DSCBlock(512, 1024, stride=(1, 1), dropout=self.dropout),
            ])
        elif output_dim == 768:
            self.init_blocks = torch.nn.Sequential(*[
                ConvBlock(3, 32, stride=(1, 1), dropout=self.dropout),
                ConvBlock(32, 64, stride=(2, 2), dropout=self.dropout),
                ConvBlock(64, 128, stride=(2, 2), dropout=self.dropout),
                ConvBlock(128, 256, stride=(2, 2), dropout=self.dropout),
                ConvBlock(256, 512, stride=(2, 1), dropout=self.dropout),
                ConvBlock(512, 512, stride=(1, 1), dropout=self.dropout),
            ])
            self.blocks = torch.nn.Sequential(*[
            # self.blocks = ModuleList([
                DSCBlock(512, 512, stride=(1, 1), dropout=self.dropout),
                DSCBlock(512, 512, stride=(1, 1), dropout=self.dropout),
                DSCBlock(512, 512, stride=(1, 1), dropout=self.dropout),
                DSCBlock(512, 768, stride=(1, 1), dropout=self.dropout),
            ])
    
    
    
    def resize_mask_to_feature_map(self, mask, target_h, target_w):
        """
        Resize a categorical mask (B, H, W) or (H, W) to (target_h, target_w),
        preserving label values, via direct nearest-neighbor indexing (no float cast).
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # (1, H, W)

        B, H, W = mask.shape
        device = mask.device

        # same indices as F.interpolate(mode='nearest'), computed as integers (exact)
        y_idx = (torch.arange(target_h, device=device) * H) // target_h
        x_idx = (torch.arange(target_w, device=device) * W) // target_w

        return mask[:, y_idx][:, :, x_idx]  # already uint8, no conversion needed
   
    def forward(self, x, images_masks=None):
        x = self.init_blocks(x)
        x = self.blocks(x)

        B, C, H, W = x.shape
        
        if images_masks is not None:
            fw_masks_2D = [self.resize_mask_to_feature_map(mask, H, W) for mask in images_masks]
            fw_masks_2D = torch.stack(fw_masks_2D, dim=0)   # (B, 1, H, W)

        else:
            fw_masks_2D = None
    
        return x, fw_masks_2D

class DSCBlock(Module):

    def __init__(self, in_, out_, stride=(2, 1), activation=ReLU, dropout=0.4):
        super(DSCBlock, self).__init__()
        
        self.activation = activation()
        self.conv1 = DepthSepConv2D(in_, out_, kernel_size=(3, 3))
        self.conv2 = DepthSepConv2D(out_, out_, kernel_size=(3, 3))
        self.conv3 = DepthSepConv2D(out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm_layer = InstanceNorm2d(out_, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout/2)

    def forward(self, x1):
        pos = random.randint(1, 3)
        x = self.conv1(x1)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)

        if pos == 3:
            x = self.dropout(x)
        # used to allow encapsulation in utils.checkpoint: (activation checkpointing)
        x = x + x1 if x.size() == x1.size() else x
        return x
    

class DepthSepConv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=None, padding=True, stride=(1, 1), dilation=(1, 1)):
        super(DepthSepConv2D, self).__init__()

        self.padding = None

        if padding:
            if padding is True:
                padding = [int((k - 1) / 2) for k in kernel_size]
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    padding_h = kernel_size[1] - 1
                    padding_w = kernel_size[0] - 1
                    self.padding = [padding_h//2, padding_h-padding_h//2, padding_w//2, padding_w-padding_w//2]
                    padding = (0, 0)

        else:
            padding = (0, 0)
        self.depth_conv = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, groups=in_channels)
        self.point_conv = Conv2d(in_channels=in_channels, out_channels=out_channels, dilation=dilation, kernel_size=(1, 1))
        self.activation = activation

    def forward(self, x):
        x = self.depth_conv(x)
        if self.padding:
            x = pad(x, self.padding)
        if self.activation:
            x = self.activation(x)
        x = self.point_conv(x)
        return x

class ConvBlock(Module):

    def __init__(self, in_, out_, stride=(1, 1), k=3, activation=ReLU, dropout=0.4):
        super(ConvBlock, self).__init__()

        self.activation = activation()
        self.conv1 = Conv2d(in_channels=in_, out_channels=out_, kernel_size=k, padding=k // 2)
        self.conv2 = Conv2d(in_channels=out_, out_channels=out_, kernel_size=k, padding=k // 2)
        self.conv3 = Conv2d(out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm_layer = InstanceNorm2d(out_, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout / 2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)
        x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        return x
    

class MixDropout(Module):
    def __init__(self, dropout_proba=0.4, dropout2d_proba=0.2):
        super(MixDropout, self).__init__()

        self.dropout = Dropout(dropout_proba)
        self.dropout2d = Dropout2d(dropout2d_proba)

    def forward(self, x):
        if random.random() < 0.5:
            return self.dropout(x)
        return self.dropout2d(x)


class SinusoidalPositionalEncoding2D(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D sinusoidal"

        self.d_model = d_model

    def forward(self, H, W):
        """
        H, W: height and width of the feature map
        returns: (1, H*W, d_model) positional encodings
        """
        pe = torch.zeros(self.d_model, H, W)

        d_model = self.d_model
        div_term = torch.exp(torch.arange(0, d_model // 2, 2) * -(math.log(10000.0) / (d_model // 2)))

        pos_w = torch.arange(W).unsqueeze(1)
        pos_h = torch.arange(H).unsqueeze(1)

        pe[0:d_model // 2:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
        pe[1:d_model // 2:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)

        pe[d_model // 2::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
        pe[d_model // 2 + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)

        pe = pe.view(d_model, H * W).transpose(0, 1).unsqueeze(0)  # (1, H*W, d_model)
        return pe
    

